extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_86,code=sm_86")
        .file("src/cuda_test.cu")
        .file("src/cuda_kernel.cu")
        .compile("libcuda_add.a");
        
    // .file("src/test_inc.cc")
    // .cpp(true)
}

