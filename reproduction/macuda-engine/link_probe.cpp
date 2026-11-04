// Offline proof of a custom llama.cpp application linked to MacCUDA's Ada stack.
// No model load or decode. Static CUDA registration may initialize the null
// driver; the launcher explicitly removes both live hardware selectors.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

static ggml_backend_reg_t (* volatile cuda_registration_symbol)() = &ggml_backend_cuda_reg;

struct Capture {
    size_t bytes_observed = 0;
    std::vector<unsigned char> buffer;
};

static bool observe(ggml_tensor * tensor, bool ask, void * opaque) {
    const char * name = ggml_get_name(tensor);
    const bool selected = std::strcmp(name, "ffn_swiglu-15") == 0 ||
                          std::strcmp(name, "ffn_swiglu-19") == 0 ||
                          std::strcmp(name, "l_out-15") == 0 ||
                          std::strcmp(name, "l_out-19") == 0;
    if (ask) return selected;
    if (selected) {
        auto & capture = *static_cast<Capture *>(opaque);
        capture.buffer.resize(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, capture.buffer.data(), 0, capture.buffer.size());
        capture.bytes_observed += capture.buffer.size();
    }
    return true;
}

int main() {
    Capture capture;
    auto params = llama_context_default_params();
    params.cb_eval = observe;
    params.cb_eval_user_data = &capture;
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    (void)model_params;
    assert(cuda_registration_symbol != nullptr);
    auto * context = ggml_init({1 << 20, nullptr, true});
    assert(context != nullptr);
    auto * tensor = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1024);
    ggml_set_name(tensor, "l_out-15");
    assert(params.cb_eval(tensor, true, params.cb_eval_user_data));
    ggml_set_name(tensor, "ffn_swiglu-19");
    assert(params.cb_eval(tensor, true, params.cb_eval_user_data));
    ggml_set_name(tensor, "unselected");
    assert(!params.cb_eval(tensor, true, params.cb_eval_user_data));
    ggml_free(context);
    std::puts("PASS: native custom application links llama.cpp + MacCUDA sm_89 archives; callback selection works.");
    std::puts("Null-device link check only; no model decode. Tensor download and model execution remain hardware tests.");
}
