//! How to use this crate
//! # Adding this as a dependency
//! ```rust, ignore
//! [dependencies]
//! wasmedge_tensorflow_interface = "^0.2.1"
//! ```
//!
//! # Bringing this into scope
//! ```rust, ignore
//! use wasmedge_tensorflow_interface;
//! ```

use std::ffi::CString;
use std::mem;
use std::str;

/// wasmedge_tensorflow host functions.
#[link(wasm_import_module = "wasmedge_tensorflow")]
extern "C" {
    pub fn wasmedge_tensorflow_create_session(model_buf: *const u8, model_buf_len: u32) -> u64;
    pub fn wasmedge_tensorflow_delete_session(context: u64);
    pub fn wasmedge_tensorflow_run_session(context: u64) -> u32;
    pub fn wasmedge_tensorflow_get_output_tensor(
        context: u64,
        output_name: *const u8,
        output_name_len: u32,
        index: u32,
    ) -> u64;
    pub fn wasmedge_tensorflow_get_tensor_len(tensor_ptr: u64) -> u32;
    pub fn wasmedge_tensorflow_get_tensor_data(tensor_ptr: u64, buf: *mut u8);
    pub fn wasmedge_tensorflow_append_input(
        context: u64,
        input_name: *const u8,
        input_name_len: u32,
        index: u32,
        dim_vec: *const u8,
        dim_cnt: u32,
        data_type: u32,
        tensor_buf: *const u8,
        tensor_buf_len: u32,
    );
    pub fn wasmedge_tensorflow_append_output(
        context: u64,
        output_name: *const u8,
        output_name_len: u32,
        index: u32,
    );
    pub fn wasmedge_tensorflow_clear_input(context: u64);
    pub fn wasmedge_tensorflow_clear_output(context: u64);
}

/// wasmedge_tensorflowlite host functions.
#[link(wasm_import_module = "wasmedge_tensorflowlite")]
extern "C" {
    pub fn wasmedge_tensorflowlite_create_session(model_buf: *const u8, model_buf_len: u32) -> u64;
    pub fn wasmedge_tensorflowlite_delete_session(context: u64);
    pub fn wasmedge_tensorflowlite_run_session(context: u64) -> u32;
    pub fn wasmedge_tensorflowlite_get_output_tensor(
        context: u64,
        output_name: *const u8,
        output_name_len: u32,
    ) -> u64;
    pub fn wasmedge_tensorflowlite_get_tensor_len(tensor_ptr: u64) -> u32;
    pub fn wasmedge_tensorflowlite_get_tensor_data(tensor_ptr: u64, buf: *mut u8);
    pub fn wasmedge_tensorflowlite_append_input(
        context: u64,
        input_name: *const u8,
        input_name_len: u32,
        tensor_buf: *const u8,
        tensor_buf_len: u32,
    );
}

/// wasmedge_image host helper functions.
#[link(wasm_import_module = "wasmedge_image")]
extern "C" {
    pub fn wasmedge_image_load_jpg_to_rgb8(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
        dst_buf: *mut u8,
    ) -> u32;
    pub fn wasmedge_image_load_jpg_to_bgr8(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
        dst_buf: *mut u8,
    ) -> u32;
    pub fn wasmedge_image_load_jpg_to_rgb32f(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
        dst_buf: *mut u8,
    ) -> u32;
    pub fn wasmedge_image_load_jpg_to_bgr32f(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
        dst_buf: *mut u8,
    ) -> u32;
    pub fn wasmedge_image_load_png_to_rgb8(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
        dst_buf: *mut u8,
    ) -> u32;
    pub fn wasmedge_image_load_png_to_bgr8(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
        dst_buf: *mut u8,
    ) -> u32;
    pub fn wasmedge_image_load_png_to_rgb32f(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
        dst_buf: *mut u8,
    ) -> u32;
    pub fn wasmedge_image_load_png_to_bgr32f(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
        dst_buf: *mut u8,
    ) -> u32;
}

/// TensorType trait. Internal only.
pub trait TensorType: Clone {
    type InnerType;
    fn val() -> u32;
    fn zero() -> Self;
}

/// Macro for mapping rust types onto tensor type. Internal only.
macro_rules! tensor_type {
    ($rust_type:ty, $type_val:expr, $zero:expr) => {
        impl TensorType for $rust_type {
            type InnerType = $rust_type;
            fn val() -> u32 {
                $type_val
            }

            fn zero() -> Self {
                $zero
            }
        }
    };
}
tensor_type!(f32, 1, 0.0);
tensor_type!(f64, 2, 0.0);
tensor_type!(i32, 3, 0);
tensor_type!(u8, 4, 0);
tensor_type!(u16, 17, 0);
tensor_type!(u32, 22, 0);
tensor_type!(u64, 23, 0);
tensor_type!(i16, 5, 0);
tensor_type!(i8, 6, 0);
tensor_type!(i64, 9, 0);
tensor_type!(bool, 10, false);

#[derive(PartialEq, Eq)]
pub enum ModelType {
    TensorFlow = 0,
    TensorFlowLite = 1,
}

/// The session structure.
pub struct Session {
    context: u64,
    model_type: ModelType,
    model_data: Vec<u8>,
}

impl Session {
    pub fn new<S: AsRef<[u8]>>(model_buf: S, mod_type: ModelType) -> Session {
        let data = Vec::from(model_buf.as_ref());
        unsafe {
            Session {
                context: if mod_type == ModelType::TensorFlow {
                    wasmedge_tensorflow_create_session(
                        data.as_slice().as_ptr() as *const u8,
                        data.len() as u32,
                    )
                } else {
                    wasmedge_tensorflowlite_create_session(
                        data.as_slice().as_ptr() as *const u8,
                        data.len() as u32,
                    )
                },
                model_type: mod_type,
                model_data: data,
            }
        }
    }

    /// Add input name, dimension, operation index, and input tensor into context.
    pub fn add_input<T: TensorType>(
        &mut self,
        name: &str,
        tensor_buf: &[T],
        shape: &[i64],
    ) -> &mut Session {
        // Parse name and operation index.
        let mut idx: u32 = 0;
        let input_name: CString;
        if self.model_type == ModelType::TensorFlow {
            let name_pair: Vec<&str> = name.split(":").collect();
            if name_pair.len() > 1 {
                idx = name_pair[1].parse().unwrap();
            }
            input_name = CString::new(name_pair[0]).expect("");
        } else {
            input_name = CString::new(name.to_string()).expect("");
        };

        // Append input tensor.
        unsafe {
            if self.model_type == ModelType::TensorFlow {
                wasmedge_tensorflow_append_input(
                    self.context,
                    input_name.as_ptr() as *const u8,
                    input_name.as_bytes().len() as u32,
                    idx,
                    shape.as_ptr() as *const u8,
                    shape.len() as u32,
                    T::val(),
                    tensor_buf.as_ptr() as *const u8,
                    (tensor_buf.len() * mem::size_of::<T>()) as u32,
                )
            } else {
                wasmedge_tensorflowlite_append_input(
                    self.context,
                    input_name.as_ptr() as *const u8,
                    input_name.as_bytes().len() as u32,
                    tensor_buf.as_ptr() as *const u8,
                    (tensor_buf.len() * mem::size_of::<T>()) as u32,
                )
            }
        }
        self
    }

    /// Add output name and operation index into context.
    pub fn add_output(&mut self, name: &str) -> &mut Session {
        // Tensorflow mode only.
        if self.model_type == ModelType::TensorFlow {
            let name_pair: Vec<&str> = name.split(":").collect();
            let output_name = CString::new(name_pair[0]).expect("");
            let idx = if name_pair.len() > 1 {
                name_pair[1].parse().unwrap()
            } else {
                0
            };
            unsafe {
                wasmedge_tensorflow_append_output(
                    self.context,
                    output_name.as_ptr() as *const u8,
                    output_name.as_bytes().len() as u32,
                    idx,
                )
            }
        }
        self
    }

    /// Clear the set input tensors.
    pub fn clear_input(&mut self) -> &mut Session {
        if self.model_type == ModelType::TensorFlow {
            unsafe {
                wasmedge_tensorflow_clear_input(self.context);
            }
        }
        self
    }

    /// Clear the set output tensors.
    pub fn clear_output(&mut self) -> &mut Session {
        if self.model_type == ModelType::TensorFlow {
            unsafe {
                wasmedge_tensorflow_clear_output(self.context);
            }
        }
        self
    }

    /// Run session.
    pub fn run(&mut self) -> &mut Session {
        unsafe {
            if self.model_type == ModelType::TensorFlow {
                wasmedge_tensorflow_run_session(self.context);
            } else {
                wasmedge_tensorflowlite_run_session(self.context);
            }
        }
        self
    }

    /// Get output tensor data by name.
    pub fn get_output<T: TensorType>(&self, name: &str) -> Vec<T> {
        // Parse name and operation index.
        let mut idx: u32 = 0;
        let output_name: CString;
        if self.model_type == ModelType::TensorFlow {
            let name_pair: Vec<&str> = name.split(":").collect();
            if name_pair.len() > 1 {
                idx = name_pair[1].parse().unwrap();
            }
            output_name = CString::new(name_pair[0]).expect("");
        } else {
            output_name = CString::new(name.to_string()).expect("");
        };

        // Get tensor data.
        let mut data: Vec<T> = Vec::new();
        unsafe {
            if self.model_type == ModelType::TensorFlow {
                let tensor = wasmedge_tensorflow_get_output_tensor(
                    self.context,
                    output_name.as_ptr() as *const u8,
                    output_name.as_bytes().len() as u32,
                    idx,
                );
                let buf_len = wasmedge_tensorflow_get_tensor_len(tensor) as usize;
                if buf_len == 0 {
                    return data;
                }
                data.resize(buf_len, T::zero());
                wasmedge_tensorflow_get_tensor_data(tensor, data.as_mut_ptr() as *mut u8);
                return data;
            } else {
                let tensor = wasmedge_tensorflowlite_get_output_tensor(
                    self.context,
                    output_name.as_ptr() as *const u8,
                    output_name.as_bytes().len() as u32,
                );
                let buf_len = wasmedge_tensorflowlite_get_tensor_len(tensor) as usize;
                if buf_len == 0 {
                    return Vec::new();
                }
                data.resize(buf_len, T::zero());
                wasmedge_tensorflowlite_get_tensor_data(tensor, data.as_mut_ptr() as *mut u8);
                return data;
            }
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            if self.model_type == ModelType::TensorFlow {
                wasmedge_tensorflow_delete_session(self.context);
            } else {
                wasmedge_tensorflowlite_delete_session(self.context);
            }
        }
        self.clear_input();
        self.clear_output();
    }
}

/// Convert JPEG image in memory into rgb u8 vector.
pub fn load_jpg_image_to_rgb8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut result_vec: Vec<u8> = vec![0; (w * h * 3) as usize];
    unsafe {
        wasmedge_image_load_jpg_to_rgb8(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
            result_vec.as_mut_ptr() as *mut u8,
        );
    }
    result_vec
}

/// Convert JPEG image in memory into bgr u8 vector.
pub fn load_jpg_image_to_bgr8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut result_vec: Vec<u8> = vec![0; (w * h * 3) as usize];
    unsafe {
        wasmedge_image_load_jpg_to_bgr8(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
            result_vec.as_mut_ptr() as *mut u8,
        );
    }
    result_vec
}

/// Convert JPEG image in memory into rgb f32 vector.
pub fn load_jpg_image_to_rgb32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    let mut result_vec: Vec<f32> = vec![0.0; (w * h * 3) as usize];
    unsafe {
        wasmedge_image_load_jpg_to_rgb32f(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
            result_vec.as_mut_ptr() as *mut u8,
        );
    }
    result_vec
}

/// Convert JPEG image in memory into bgr f32 vector.
pub fn load_jpg_image_to_bgr32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    let mut result_vec: Vec<f32> = vec![0.0; (w * h * 3) as usize];
    unsafe {
        wasmedge_image_load_jpg_to_bgr32f(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
            result_vec.as_mut_ptr() as *mut u8,
        );
    }
    result_vec
}

/// Convert PNG image in memory into rgb u8 vector.
pub fn load_png_image_to_rgb8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut result_vec: Vec<u8> = vec![0; (w * h * 3) as usize];
    unsafe {
        wasmedge_image_load_png_to_rgb8(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
            result_vec.as_mut_ptr() as *mut u8,
        );
    }
    result_vec
}

/// Convert PNG image in memory into bgr u8 vector.
pub fn load_png_image_to_bgr8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut result_vec: Vec<u8> = vec![0; (w * h * 3) as usize];
    unsafe {
        wasmedge_image_load_png_to_bgr8(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
            result_vec.as_mut_ptr() as *mut u8,
        );
    }
    result_vec
}

/// Convert PNG image in memory into rgb f32 vector.
pub fn load_png_image_to_rgb32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    let mut result_vec: Vec<f32> = vec![0.0; (w * h * 3) as usize];
    unsafe {
        wasmedge_image_load_png_to_rgb32f(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
            result_vec.as_mut_ptr() as *mut u8,
        );
    }
    result_vec
}

/// Convert PNG image in memory into bgr f32 vector.
pub fn load_png_image_to_bgr32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    let mut result_vec: Vec<f32> = vec![0.0; (w * h * 3) as usize];
    unsafe {
        wasmedge_image_load_png_to_bgr32f(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
            result_vec.as_mut_ptr() as *mut u8,
        );
    }
    result_vec
}
