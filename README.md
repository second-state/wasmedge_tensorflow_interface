# SSVM Tensorflow Interface

A Rust library that provides Rust to WebAssembly developers with syntax for using tensorflow functionality when their Wasm is being executed on [SecondState's SSVM](https://github.com/second-state/SSVM).

From a high-level overview here, we are essentially building a tensorflow interface that will allow the native operating system (which SSVM is running on) to play a part in the runtime execution. Specifically, play a part in using tensorflow with graphs and input and output tensors as part of Wasm execution. 

# How to use this library

## Rust dependency

Developers will add the [`ssvm_interface_interface` crate](https://crates.io/crates/ssvm_interface_interface) as a dependency to their `Rust -> Wasm` applications. For example, add the following line to the application's `Cargo.toml` file.
```
[dependencies]
ssvm_tensorflow_interface = "^0.1.0"
```

# Crates.io

The official crate is available at [crates.io](https://crates.io/crates/ssvm_tensorflow_interface).
