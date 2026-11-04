// Real MacCUDA throughput, with experiment sampling and independent sequences.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
struct Sampler {
    struct Candidate {float value;llama_token id;};
    std::vector<unsigned char> seen;
    std::mt19937 rng;
    Sampler(int n,uint32_t seed):seen(n,0),rng(seed){}
    llama_token sample(const float * logits) {
        std::array<Candidate,20> top;
        for(auto & c:top)c={-INFINITY,0};
        int worst=0;
        for(size_t i=0;i<seen.size();++i){
            float x=logits[i]-1.5f*seen[i];
            if(!std::isfinite(x)){std::fprintf(stderr,"Nonfinite logit\n");std::exit(8);}
            if(x>top[worst].value){
                top[worst]={x,static_cast<llama_token>(i)};
                worst=0;for(int j=1;j<20;++j)if(top[j].value<top[worst].value)worst=j;
            }
        }
        std::sort(top.begin(),top.end(),[](auto a,auto b){return a.value>b.value;});
        std::array<double,20> probs;
        double sum=0;
        for(int j=0;j<20;++j){probs[j]=std::exp((top[j].value-top[0].value)/.6);sum+=probs[j];}
        double kept=0;int n=0;
        do{kept+=probs[n++];}while(n<20 && kept<.95*sum);
        double draw=std::uniform_real_distribution<double>(0,kept)(rng);
        int chosen=0;
        while(chosen<n-1 && draw>probs[chosen])draw-=probs[chosen++];
        llama_token token=top[chosen].id;seen[token]=1;return token;
    }
};
struct Batch {
    llama_batch b;
    explicit Batch(int n):b(llama_batch_init(n,0,1)){b.n_tokens=0;}
    ~Batch(){llama_batch_free(b);}
    void add(llama_token t,int pos,int seq,bool logits){
        int i=b.n_tokens++;b.token[i]=t;b.pos[i]=pos;b.n_seq_id[i]=1;b.seq_id[i][0]=seq;b.logits[i]=logits;
    }
};
struct Case {int batch,cap;};
int main(int argc,char ** argv){
    if(argc!=5){std::fprintf(stderr,"usage: inference_bench model.gguf prompt-dir output-dir batch:tokens[,batch:tokens...]\n");return 2;}
    std::filesystem::path output=argv[3];
    if(std::filesystem::exists(output)){std::fprintf(stderr,"Output exists\n");return 2;}
    std::filesystem::create_directories(output);
    std::vector<Case> cases;std::string specification=argv[4];
    for(size_t start=0;start<specification.size();){
        size_t end=specification.find(',',start);if(end==std::string::npos)end=specification.size();
        auto part=specification.substr(start,end-start);size_t colon=part.find(':');
        if(colon==std::string::npos)return 2;
        Case c{std::stoi(part.substr(0,colon)),std::stoi(part.substr(colon+1))};
        if(c.batch<1||c.batch>32||c.cap<1||c.cap>32768)return 2;
        cases.push_back(c);start=end+1;
    }
    ggml_backend_load_all();auto dev=ggml_backend_dev_by_name("CUDA0");
    if(!dev||!std::getenv("TINYNV_SOCKET")||std::string(ggml_backend_dev_description(dev)).find("AD102")==std::string::npos){
        std::fprintf(stderr,"Real AD102 CUDA0 required\n");return 3;
    }
    ggml_backend_dev_t devices[]={dev,nullptr};
    auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=99;mp.check_tensors=true;
    auto load_start=Clock::now();
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
    if(!model)return 4;
    const double load_time=seconds(load_start);
    const auto * vocab=llama_model_get_vocab(model.get());const int nv=llama_vocab_n_tokens(vocab);
    std::ofstream jsonl(output/"cases.jsonl");
    for(auto test:cases){
        auto cp=llama_context_default_params();
        cp.n_ctx=test.batch*(test.cap+2048);cp.n_seq_max=test.batch;
        cp.n_batch=2048;cp.n_ubatch=512;cp.n_threads=8;cp.n_threads_batch=8;
        cp.type_k=GGML_TYPE_BF16;cp.type_v=GGML_TYPE_BF16;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;cp.kv_unified=false;cp.offload_kqv=true;
        cp.cb_eval=nullptr;cp.cb_eval_user_data=nullptr;
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
        if(!ctx)return 4;
        std::vector<std::vector<llama_token>> prompts(test.batch),generated(test.batch);
        std::vector<Sampler> samplers;std::vector<bool> done(test.batch,false),eos(test.batch,false);
        for(int s=0;s<test.batch;++s){
            std::ifstream input(std::filesystem::path(argv[2])/(std::to_string(s)+".txt"),std::ios::binary);
            std::string text((std::istreambuf_iterator<char>(input)),{});
            if(!input||text.empty())return 2;
            int n=-llama_tokenize(vocab,text.data(),text.size(),nullptr,0,false,true);
            if(n<1||n>2048)return 2;
            prompts[s].resize(n);
            if(llama_tokenize(vocab,text.data(),text.size(),prompts[s].data(),n,false,true)!=n)return 2;
            samplers.emplace_back(nv,1729+s);
        }
        std::fprintf(stderr,"BENCH START batch=%d cap=%d BF16 weights/KV, flash-attention on\n",test.batch,test.cap);
        auto begin=Clock::now();auto progress=begin;size_t previous=0;double prefill=0;
        for(int s=0;s<test.batch;++s){
            Batch b(prompts[s].size());
            for(size_t i=0;i<prompts[s].size();++i)b.add(prompts[s][i],i,s,i+1==prompts[s].size());
            if(llama_decode(ctx.get(),b.b)!=0)return 5;
            auto token=samplers[s].sample(llama_get_logits_ith(ctx.get(),-1));
            if(llama_vocab_is_eog(vocab,token)){done[s]=true;eos[s]=true;}
            else{generated[s].push_back(token);done[s]=test.cap==1;}
        }
        llama_synchronize(ctx.get());prefill=seconds(begin);
        Batch step(test.batch);
        while(!std::all_of(done.begin(),done.end(),[](bool x){return x;})){
            step.b.n_tokens=0;std::vector<int> slots;
            for(int s=0;s<test.batch;++s)if(!done[s]){
                slots.push_back(s);step.add(generated[s].back(),prompts[s].size()+generated[s].size()-1,s,true);
            }
            if(llama_decode(ctx.get(),step.b)!=0)return 5;
            for(size_t i=0;i<slots.size();++i){
                int s=slots[i];auto token=samplers[s].sample(llama_get_logits_ith(ctx.get(),i));
                if(llama_vocab_is_eog(vocab,token)){done[s]=true;eos[s]=true;}
                else{generated[s].push_back(token);if(static_cast<int>(generated[s].size())>=test.cap)done[s]=true;}
            }
            size_t count=0;int maximum=0,active=0;
            for(int s=0;s<test.batch;++s){count+=generated[s].size();maximum=std::max(maximum,static_cast<int>(generated[s].size()));active+=!done[s];}
            if(seconds(progress)>=20){
                std::fprintf(stderr,"BENCH PROGRESS batch=%d elapsed=%.1f tokens=%zu max_sequence=%d active=%d window_tps=%.1f\n",
                    test.batch,seconds(begin),count,maximum,active,(count-previous)/seconds(progress));
                std::fflush(stderr);progress=Clock::now();previous=count;
            }
            if(seconds(begin)>1200){std::fprintf(stderr,"BENCH ABORT: per-case wall limit exceeded\n");return 9;}
        }
        llama_synchronize(ctx.get());double wall=seconds(begin);size_t count=0;
        for(auto & g:generated)count+=g.size();
        jsonl << "{\"batch\":" << test.batch << ",\"max_new_tokens\":" << test.cap
              << ",\"output_tokens\":" << count << ",\"wall_seconds_including_prefill\":" << wall
              << ",\"prefill_seconds_including_first_sample\":" << prefill
              << ",\"aggregate_output_tps_including_prefill\":" << count/wall
              << ",\"model_load_seconds\":" << load_time << ",\"flash_attention\":true,\"kv_dtype\":\"BF16\",\"sequences\":[";
        for(int s=0;s<test.batch;++s){
            if(s)jsonl << ',';
            jsonl << "{\"slot\":" << s << ",\"prompt_tokens\":" << prompts[s].size() << ",\"output_tokens\":" << generated[s].size()
                  << ",\"finish_reason\":\"" << (eos[s]?"stop":"length") << "\",\"token_ids\":[";
            for(size_t j=0;j<generated[s].size();++j)jsonl << (j?",":"") << generated[s][j];
            jsonl << "]}";
        }
        jsonl << "]}\n";jsonl.flush();
        std::fprintf(stderr,"BENCH RESULT batch=%d cap=%d tokens=%zu wall=%.3f aggregate_tps=%.3f\n",test.batch,test.cap,count,wall,count/wall);
        std::fflush(stderr);
    }
    std::ofstream(output/"completed.txt") << "completed\n";
    return 0;
}
