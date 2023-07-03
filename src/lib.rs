//! How to use this crate
//! # Adding this as a dependency
//! ```rust, ignore
//! [dependencies]
//! wasmedge_tensorflow_interface = "^0.3.0"
//! ```
//!
//! # Bringing this into scope
//! ```rust, ignore
//! use wasmedge_tensorflow_interface;
//! ```

mod generated_tf;
mod generated_tflite;
mod generated_img;
pub use generated_tf::*;
pub use generated_tflite::*;
pub use generated_img::*;
use std::mem;

// TensorType trait. Internal only.
pub trait TensorType: Clone {
    type InnerType;
    fn val() -> u32;
    fn zero() -> Self;
}

// Macro for mapping rust types onto tensor type. Internal only.
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
tensor_type!(f32, 1, 0.0f32);
tensor_type!(f64, 2, 0.0f64);
tensor_type!(i32, 3, 0);
tensor_type!(u8, 4, 0);
tensor_type!(u16, 17, 0);
tensor_type!(u32, 22, 0);
tensor_type!(u64, 23, 0);
tensor_type!(i16, 5, 0);
tensor_type!(i8, 6, 0);
tensor_type!(i64, 9, 0);
tensor_type!(bool, 10, false);

// The TensorFlow session structure.
pub struct TFSession {
    context: generated_tf::Session,
}

impl TFSession {
    pub fn new<S: AsRef<[u8]>>(model_buf: S) -> TFSession {
        let data = Vec::from(model_buf.as_ref());
        unsafe {
            TFSession {
                context: generated_tf::create_session(&data).unwrap(),
            }
        }
    }

    pub fn new_from_saved_model(model_path: &str, tags: &[&str]) -> TFSession {
        unsafe {
            TFSession {
                context: generated_tf::create_session_saved_model(model_path, tags).unwrap(),
            }
        }
    }

    // Add input name, dimension, operation index, and input tensor into context.
    pub fn add_input<T: TensorType>(
        &mut self,
        name: &str,
        tensor_buf: &[T],
        shape: &[u64],
    ) -> &mut TFSession {
        unsafe {
            let raw_buf = std::slice::from_raw_parts(
                (tensor_buf as *const [T]).cast(),
                tensor_buf.len() * mem::size_of::<T::InnerType>()
            );
            generated_tf::append_input(
                self.context,
                name,
                shape,
                T::val(),
                raw_buf,
            ).unwrap();
        }
        self
    }

    // Add output name and operation index into context.
    pub fn add_output(&mut self, name: &str) -> &mut TFSession {
        unsafe {
            generated_tf::append_output(self.context, name).unwrap();
        }
        self
    }

    // Clear the set input tensors.
    pub fn clear_input(&mut self) -> &mut TFSession {
        unsafe {
            generated_tf::clear_input(self.context).unwrap();
        }
        self
    }

    // Clear the set output tensors.
    pub fn clear_output(&mut self) -> &mut TFSession {
        unsafe {
            generated_tf::clear_output(self.context).unwrap();
        }
        self
    }

    // Run session.
    pub fn run(&mut self) -> &mut TFSession {
        unsafe {
            generated_tf::run_session(self.context).unwrap();
        }
        self
    }

    // Get output tensor data by name.
    pub fn get_output<T: TensorType>(&self, name: &str) -> Vec<T> {
        // Get tensor data.
        let mut data: Vec<T> = Vec::new();
        unsafe {
            let tensor = generated_tf::get_output_tensor(self.context, name).unwrap();
            let buf_len = generated_tf::get_tensor_len(self.context, tensor).unwrap() as usize;
            if buf_len == 0 {
                return data;
            }
            data.resize(buf_len / mem::size_of::<T::InnerType>(), T::zero());
            generated_tf::get_tensor_data(self.context, tensor, data.as_mut_ptr() as *mut u8, buf_len as u32).unwrap();
            return data;
        }
    }
}

impl Drop for TFSession {
    fn drop(&mut self) {
        unsafe {
            generated_tf::delete_session(self.context).unwrap();
        }
    }
}

// The TensorFlow-Lite session structure.
pub struct TFLiteSession {
    context: generated_tflite::Session,
}

impl TFLiteSession {
    pub fn new<S: AsRef<[u8]>>(model_buf: S) -> TFLiteSession {
        let data = Vec::from(model_buf.as_ref());
        unsafe {
            TFLiteSession {
                context: generated_tflite::create_session(&data).unwrap(),
            }
        }
    }

    // Add input name, dimension, operation index, and input tensor into context.
    pub fn add_input<T: TensorType>(
        &mut self,
        name: &str,
        tensor_buf: &[T],
    ) -> &mut TFLiteSession {
        unsafe {
            let raw_buf = std::slice::from_raw_parts(
                (tensor_buf as *const [T]).cast(),
                tensor_buf.len() * mem::size_of::<T::InnerType>()
            );
            generated_tflite::append_input(
                self.context,
                name,
                raw_buf,
            ).unwrap();
        }
        self
    }

    // Run session.
    pub fn run(&mut self) -> &mut TFLiteSession {
        unsafe {
            generated_tflite::run_session(self.context).unwrap();
        }
        self
    }

    // Get output tensor data by name.
    pub fn get_output<T: TensorType>(&self, name: &str) -> Vec<T> {
        // Get tensor data.
        let mut data: Vec<T> = Vec::new();
        unsafe {
            let tensor = generated_tflite::get_output_tensor(self.context, name).unwrap();
            let buf_len = generated_tflite::get_tensor_len(self.context, tensor).unwrap() as usize;
            if buf_len == 0 {
                return data;
            }
            data.resize(buf_len / mem::size_of::<T::InnerType>(), T::zero());
            generated_tflite::get_tensor_data(self.context, tensor, data.as_mut_ptr() as *mut u8, buf_len as u32).unwrap();
            return data;
        }
    }
}

impl Drop for TFLiteSession {
    fn drop(&mut self) {
        unsafe {
            generated_tflite::delete_session(self.context).unwrap();
        }
    }
}

// The Image functions.
// Convert JPEG image in memory into rgb u8 vector.
pub fn load_jpg_image_to_rgb8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut result_vec: Vec<u8> = vec![0; (w * h * 3) as usize];
    unsafe {
        generated_img::load_jpg(
            img_buf,
            w,
            h,
            generated_img::WASMEDGE_IMAGE_RAW_TYPE_RGB8,
            result_vec.as_mut_ptr() as *mut u8,
            result_vec.len() as u32,
        ).unwrap();
    }
    result_vec
}

// Convert JPEG image in memory into bgr u8 vector.
pub fn load_jpg_image_to_bgr8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut result_vec: Vec<u8> = vec![0; (w * h * 3) as usize];
    unsafe {
        generated_img::load_jpg(
            img_buf,
            w,
            h,
            generated_img::WASMEDGE_IMAGE_RAW_TYPE_BGR8,
            result_vec.as_mut_ptr() as *mut u8,
            result_vec.len() as u32,
        ).unwrap();
    }
    result_vec
}

// Convert JPEG image in memory into rgb f32 vector.
pub fn load_jpg_image_to_rgb32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    let mut result_vec: Vec<f32> = vec![0.0; (w * h * 3) as usize];
    unsafe {
        generated_img::load_jpg(
            img_buf,
            w,
            h,
            generated_img::WASMEDGE_IMAGE_RAW_TYPE_RGB32F,
            result_vec.as_mut_ptr() as *mut u8,
            (result_vec.len() * mem::size_of::<f32>()) as u32,
        ).unwrap();
    }
    result_vec
}

// Convert JPEG image in memory into bgr f32 vector.
pub fn load_jpg_image_to_bgr32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    let mut result_vec: Vec<f32> = vec![0.0; (w * h * 3) as usize];
    unsafe {
        generated_img::load_jpg(
            img_buf,
            w,
            h,
            generated_img::WASMEDGE_IMAGE_RAW_TYPE_BGR32F,
            result_vec.as_mut_ptr() as *mut u8,
            (result_vec.len() * mem::size_of::<f32>()) as u32,
        ).unwrap();
    }
    result_vec
}

// Convert PNG image in memory into rgb u8 vector.
pub fn load_png_image_to_rgb8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut result_vec: Vec<u8> = vec![0; (w * h * 3) as usize];
    unsafe {
        generated_img::load_png(
            img_buf,
            w,
            h,
            generated_img::WASMEDGE_IMAGE_RAW_TYPE_RGB8,
            result_vec.as_mut_ptr() as *mut u8,
            result_vec.len() as u32,
        ).unwrap();
    }
    result_vec
}

// Convert PNG image in memory into bgr u8 vector.
pub fn load_png_image_to_bgr8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut result_vec: Vec<u8> = vec![0; (w * h * 3) as usize];
    unsafe {
        generated_img::load_png(
            img_buf,
            w,
            h,
            generated_img::WASMEDGE_IMAGE_RAW_TYPE_BGR8,
            result_vec.as_mut_ptr() as *mut u8,
            result_vec.len() as u32,
        ).unwrap();
    }
    result_vec
}

// Convert PNG image in memory into rgb f32 vector.
pub fn load_png_image_to_rgb32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    let mut result_vec: Vec<f32> = vec![0.0; (w * h * 3) as usize];
    unsafe {
        generated_img::load_png(
            img_buf,
            w,
            h,
            generated_img::WASMEDGE_IMAGE_RAW_TYPE_RGB32F,
            result_vec.as_mut_ptr() as *mut u8,
            (result_vec.len() * mem::size_of::<f32>()) as u32,
        ).unwrap();
    }
    result_vec
}

// Convert PNG image in memory into bgr f32 vector.
pub fn load_png_image_to_bgr32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    let mut result_vec: Vec<f32> = vec![0.0; (w * h * 3) as usize];
    unsafe {
        generated_img::load_png(
            img_buf,
            w,
            h,
            generated_img::WASMEDGE_IMAGE_RAW_TYPE_BGR32F,
            result_vec.as_mut_ptr() as *mut u8,
            (result_vec.len() * mem::size_of::<f32>()) as u32,
        ).unwrap();
    }
    result_vec
}
