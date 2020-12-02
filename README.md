# SSVM Tensorflow Interface

A Rust library that provides Rust to WebAssembly developers with syntax for using tensorflow functionality when their Wasm is being executed on [SecondState's SSVM](https://github.com/second-state/SSVM).

From a high-level overview here, we are essentially building a tensorflow interface that will allow the native operating system (which SSVM is running on) to play a part in the runtime execution. Specifically, play a part in using tensorflow with graphs and input and output tensors as part of Wasm execution. 

# How to use this library

## Rust dependency

Developers will add the [`ssvm_interface_interface` crate](https://crates.io/crates/ssvm_interface_interface) as a dependency to their `Rust -> Wasm` applications. For example, add the following line to the application's `Cargo.toml` file.
```
[dependencies]
ssvm_tensorflow_interface = "^0.1.3"
```

Developers will bring the functions of `ssvm_tensorflow_interface` into scope within their `Rust -> Wasm` application's code. For example, adding the following code to the top of their `main.rs
```
use ssvm_process_interface;
```

## Image Loading And Conversion

```rust
let mut file_img = File::open("sample.jpg").unwrap();
let mut img_buf = Vec::new();
file_img.read_to_end(&mut img_buf).unwrap();
let flat_img = ssvm_tensorflow_interface::load_jpg_image_to_rgb32f(&img_buf, 224, 224);
// The flat_img is a vec<f32> which contains normalized image in rgb32f format and resized to 224x224.
```

## Create Session

```rust
// The mod_buf is a vec<u8> which contains model data.
let mut session = ssvm_tensorflow_interface::Session::new(&mod_buf, ssvm_tensorflow_interface::ModelType::TensorFlow);
```

Or use the `ssvm_tensorflow_interface::ModelType::TensorFlowLite` to specify the `tflite` models.

## Prepare Input Tensors

```rust
// The flat_img is a vec<f32> which contains normalized image in rgb32f format.
session.add_input("input", &flat_img, &[1, 224, 224, 3])
       .add_output("MobilenetV2/Predictions/Softmax");
```

## Run TensorFlow Models

```rust
session.run();
```

## Convert Output Tensors

```rust
let res_vec: Vec<f32> = session.get_output("MobilenetV2/Predictions/Softmax");
```

## Build And Execution

```bash
$ cargo build --target=wasm32-wasi
```

The output WASM file will be at `target/wasm32-wasi/debug/` or `target/wasm32-wasi/release`.
Please refer [SSVM with tensorflow extension](https://github.com/second-state/ssvm-tensorflow) for WASM execution.

# Crates.io

The official crate is available at [crates.io](https://crates.io/crates/ssvm_tensorflow_interface).
