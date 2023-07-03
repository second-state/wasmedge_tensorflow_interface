# WasmEdge Tensorflow Interface

A Rust library that provides Rust to WebAssembly developers with syntax for using tensorflow functionality when their Wasm is being executed on [WasmEdge](https://github.com/WasmEdge/WasmEdge).

From a high-level overview here, we are essentially building a tensorflow interface that will allow the native operating system (which WasmEdge is running on) to play a part in the runtime execution. Specifically, play a part in inferring a TensorFlow or TensorFlow-Lite with graphs and input and output tensors as part of Wasm execution.

## How to use this library

### Rust dependency

Developers will add the [`wasmedge_tensorflow_interface` crate](https://crates.io/crates/wasmedge_tensorflow_interface) as a dependency to their `Rust -> Wasm` applications. For example, add the following line to the application's `Cargo.toml` file.

```toml
[dependencies]
wasmedge_tensorflow_interface = "0.3.0"
```

Developers will bring the functions of `wasmedge_tensorflow_interface` into scope within their `Rust -> Wasm` application's code. For example, adding the following code to the top of their `main.rs`

```rust
use wasmedge_tensorflow_interface;
```

### Image Loading And Conversion

In this crate, we provide several functions to decode and convert images into tensors by using the `WasmEdge-Image` host functions.

For decoding the `JPEG` images, there are:

```rust
// Function to decode JPEG from buffer and resize to RGB8 format.
pub fn load_jpg_image_to_rgb8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8>
// Function to decode JPEG from buffer and resize to BGR8 format.
pub fn load_jpg_image_to_bgr8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8>
// Function to decode JPEG from buffer and resize to RGB32F format.
pub fn load_jpg_image_to_rgb32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32>
// Function to decode JPEG from buffer and resize to BGR32F format.
pub fn load_jpg_image_to_rgb32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32>
```

For decoding the `PNG` images, there are:

```rust
// Function to decode PNG from buffer and resize to RGB8 format.
pub fn load_png_image_to_rgb8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8>
// Function to decode PNG from buffer and resize to BGR8 format.
pub fn load_png_image_to_bgr8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8>
// Function to decode PNG from buffer and resize to RGB32F format.
pub fn load_png_image_to_rgb32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32>
// Function to decode PNG from buffer and resize to BGR32F format.
pub fn load_png_image_to_rgb32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32>
```

Developers can load, decode, and resize image as following:

```rust
let mut file_img = File::open("sample.jpg").unwrap();
let mut img_buf = Vec::new();
file_img.read_to_end(&mut img_buf).unwrap();
let flat_img = wasmedge_tensorflow_interface::load_jpg_image_to_rgb32f(&img_buf, 224, 224);
// The flat_img is a vec<f32> which contains normalized image in rgb32f format and resized to 224x224.
```

For using the above funcions in WASM and executing in WasmEdge, users should install the [WasmEdge-Image plug-in](https://wasmedge.org/docs/contribute/source/plugin/image).

### Inferring TensorFlow And TensorFlow-Lite Models

#### Create Session

First, developers should create a session to load the TensorFlow or TensorFlow-Lite model.

```rust
// The mod_buf is a vec<u8> which contains the model data.
let mut session = wasmedge_tensorflow_interface::TFSession::new(&mod_buf);
```

The above function is create the session for TensorFlow frozen models. Developers can use the `new_from_saved_model` function to create from saved-models:

```rust
// The mod_path is a &str which is the path to saved-model directory.
// The second argument is the list of tags.
let mut session = wasmedge_tensorflow_interface::TFSession::new_from_saved_model(model_path, &["serve"]);
```

Or use the `TFLiteSession` to create a session for inferring the `tflite` models.

```rust
// The mod_buf is a vec<u8> which contains the model data.
let mut session = wasmedge_tensorflow_interface::TFLiteSession::new(&mod_buf);
```

For using the `TFSession` struct and executing in WasmEdge, users should install the [WasmEdge-TensorFlow plug-in with dependencies](https://wasmedge.org/docs/contribute/source/plugin/tensorflow).

For using the `TFLiteSession` struct and executing in WasmEdge, users should install the [WasmEdge-TensorFlowLite plug-in with dependencies](https://wasmedge.org/docs/contribute/source/plugin/tensorflowlite).

#### Prepare Input Tensors

```rust
// The flat_img is a vec<f32> which contains normalized image in rgb32f format.
session.add_input("input", &flat_img, &[1, 224, 224, 3])
       .add_output("MobilenetV2/Predictions/Softmax");
```

#### Run TensorFlow Models

```rust
session.run();
```

#### Convert Output Tensors

```rust
let res_vec: Vec<f32> = session.get_output("MobilenetV2/Predictions/Softmax");
```

#### Build And Execution

```bash
cargo build --target=wasm32-wasi
```

The output WASM file will be at `target/wasm32-wasi/debug/` or `target/wasm32-wasi/release`.

Please refer to the [WasmEdge installation](https://wasmedge.org/docs/develop/build-and-run/install) to install WasmEdge with the necessary plug-ins, and [WasmEdge CLI](https://wasmedge.org/docs/develop/build-and-run/cli) WASM execution.

## Crates.io

The official crate is available at [crates.io](https://crates.io/crates/wasmedge_tensorflow_interface).
