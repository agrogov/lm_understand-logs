// Forward-only Qwen3.5 probe: explicit backend, prompt activations, greedy decode.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
struct Capture {
    bool enabled=true;
    bool failed=false;
    bool require_cuda=false;
    std::filesystem::path directory;
    std::map<std::string,int> counts;
};
static bool observe(ggml_tensor * t, bool ask, void * opaque) {
    auto & c=*static_cast<Capture *>(opaque);
    const std::string name=ggml_get_name(t);
    const bool chosen=c.enabled && (name=="l_out-15" || name=="l_out-19" ||
                                    name=="ffn_swiglu-15" || name=="ffn_swiglu-19");
    if (ask) return chosen;
    if (!chosen) return true;
    const std::string buffer_name=ggml_backend_buffer_name(t->buffer);
    if (c.require_cuda && buffer_name.find("CUDA0")==std::string::npos) {
        std::fprintf(stderr,"Selected activation is not on CUDA0: %s (%s)\n",name.c_str(),buffer_name.c_str());
        c.failed=true; return false;
    }
    if (t->type!=GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        std::fprintf(stderr,"Capture requires contiguous F32: %s\n",name.c_str()); c.failed=true; return false;
    }
    std::vector<float> values(ggml_nelements(t));
    ggml_backend_tensor_get(t,values.data(),0,ggml_nbytes(t));
    if (!std::all_of(values.begin(),values.end(),[](float x){return std::isfinite(x);})) {
        std::fprintf(stderr,"Nonfinite activation: %s\n",name.c_str()); c.failed=true; return false;
    }
    std::ofstream raw(c.directory/(name+".f32"),std::ios::binary);
    raw.write(reinterpret_cast<const char *>(values.data()),ggml_nbytes(t));
    std::ofstream meta(c.directory/(name+".json"));
    meta << "{\"dtype\":\"float32\",\"backend_buffer\":\"" << buffer_name << "\",\"ggml_ne\":[" << t->ne[0] << ',' << t->ne[1] << ',' << t->ne[2] << ',' << t->ne[3]
         << "],\"ggml_nb\":[" << t->nb[0] << ',' << t->nb[1] << ',' << t->nb[2] << ',' << t->nb[3] << "]}\n";
    if (!raw || !meta) { c.failed=true; return false; }
    ++c.counts[name];
    return true;
}
int main(int argc,char ** argv) {
    const bool cpu=argc>1 && std::strcmp(argv[1],"--cpu")==0;
    const int offset=cpu?2:1;
    if (argc!=offset+3) {
        std::fprintf(stderr,"usage: model_probe [--cpu] model.gguf raw-prompt.txt output-dir\n"); return 2;
    }
    std::ifstream input(argv[offset+1],std::ios::binary);
    std::string prompt((std::istreambuf_iterator<char>(input)),{});
    if (!input || prompt.empty()) return 2;
    const std::filesystem::path out=argv[offset+2];
    if (std::filesystem::exists(out)) { std::fprintf(stderr,"Output exists; use a new directory\n"); return 2; }
    std::filesystem::create_directories(out);
    ggml_backend_load_all();
    auto dev=cpu?ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU):ggml_backend_dev_by_name("CUDA0");
    if (!dev) { std::fprintf(stderr,"Required backend unavailable\n"); return 3; }
    const std::string description=ggml_backend_dev_description(dev);
    if (!cpu && (!std::getenv("TINYNV_SOCKET") || description.find("AD102")==std::string::npos)) {
        std::fprintf(stderr,"Refusing non-AD102/null/unselected GPU: %s\n",description.c_str()); return 3;
    }
    ggml_backend_dev_t devices[]={dev,nullptr};
    auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=cpu?0:99;mp.check_tensors=true;
    auto start=Clock::now();
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[offset],mp),llama_model_free);
    if (!model) return 4;
    const double load_s=elapsed(start);
    const auto * vocab=llama_model_get_vocab(model.get());
    int n=-llama_tokenize(vocab,prompt.data(),prompt.size(),nullptr,0,false,true);
    if (n<1 || n>256) { std::fprintf(stderr,"Expected short diagnostic prompt; got %d tokens\n",n); return 4; }
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab,prompt.data(),prompt.size(),tokens.data(),n,false,true)!=n) return 4;
    Capture capture;capture.directory=out;capture.require_cuda=!cpu;
    auto cp=llama_context_default_params();
    cp.n_ctx=512;cp.n_batch=256;cp.n_ubatch=256;cp.n_threads=8;cp.n_threads_batch=8;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=!cpu;
    cp.cb_eval=observe;cp.cb_eval_user_data=&capture;
    std::unique_ptr<llama_context,decltype(&llama_free)> context(llama_init_from_model(model.get(),cp),llama_free);
    if (!context) return 4;
    start=Clock::now();
    if (llama_decode(context.get(),llama_batch_get_one(tokens.data(),tokens.size()))!=0 || capture.failed) return 5;
    llama_synchronize(context.get());
    const double prefill_s=elapsed(start);
    if (capture.counts.size()!=4 || !std::all_of(capture.counts.begin(),capture.counts.end(),[](const auto & p){return p.second==1;})) {
        std::fprintf(stderr,"Expected exactly four prompt captures, got %zu\n",capture.counts.size()); return 5;
    }
    const int nv=llama_vocab_n_tokens(vocab);
    const float * logits=llama_get_logits_ith(context.get(),-1);
    std::ofstream logits_file(out/"last-prompt-logits.f32",std::ios::binary);
    logits_file.write(reinterpret_cast<const char *>(logits),nv*sizeof(float));logits_file.close();
    capture.enabled=false;
    std::vector<llama_token> generated;
    start=Clock::now();
    for (int i=0;i<32;++i) {
        logits=llama_get_logits_ith(context.get(),-1);
        const llama_token next=std::max_element(logits,logits+nv)-logits;
        if (llama_vocab_is_eog(vocab,next)) break;
        generated.push_back(next);
        llama_token t=next;
        if (llama_decode(context.get(),llama_batch_get_one(&t,1))!=0) return 6;
    }
    llama_synchronize(context.get());
    const double decode_s=elapsed(start);
    std::ofstream result(out/"result.json");
    result << "{\"backend\":\"" << (cpu?"CPU":"CUDA0") << "\",\"device\":\"" << description
           << "\",\"prompt_tokens\":" << n << ",\"vocab_size\":" << nv
           << ",\"model_load_seconds\":" << load_s << ",\"prefill_with_capture_seconds\":" << prefill_s
           << ",\"greedy_decode_seconds\":" << decode_s << ",\"generated_tokens\":" << generated.size()
           << ",\"greedy_tokens_per_second\":" << generated.size()/decode_s
           << ",\"capture_tensors\":" << capture.counts.size() << ",\"token_ids\":[";
    for (size_t i=0;i<generated.size();++i) result << (i?",":"") << generated[i];
    result << "]}\n"; result.close();
    std::printf("PASS: %s, %d prompt tokens, %zu captured tensors, %zu greedy tokens, %.2f tokens/s\n",
                description.c_str(),n,capture.counts.size(),generated.size(),generated.size()/decode_s);
    return 0;
}
