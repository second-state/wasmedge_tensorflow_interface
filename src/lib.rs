//! How to use this crate
//! # Adding this as a dependency
//! ```rust, ignore
//! [dependencies]
//! ssvm_tensorflow_interface = "^0.1.2"
//! ```
//!
//! # Bringing this into scope
//! ```rust, ignore
//! use ssvm_tensorflow_interface;
//! ```

use std::ffi::CString;
use std::mem;
use std::str;

/// ssvm_tensorflow host functions.
#[link(wasm_import_module = "ssvm_tensorflow")]
extern "C" {
    pub fn ssvm_tensorflow_load_jpg_to_rgb8(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_jpg_to_bgr8(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_jpg_to_rgb32f(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_jpg_to_bgr32f(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_png_to_rgb8(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_png_to_bgr8(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_png_to_rgb32f(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_png_to_bgr32f(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_exec_model(
        model_buf: *const u8,
        model_buf_len: u32,
        out_tensors: *mut u8,
    ) -> u32;
    pub fn ssvm_tensorflow_alloc_tensor(
        dim_vec: *const u8,
        dim_cnt: u32,
        data_type: u32,
        tensor_buf: *const u8,
        tensor_buf_len: u32,
    ) -> u64;
    pub fn ssvm_tensorflow_delete_tensor(tensor_ptr: u64);
    pub fn ssvm_tensorflow_get_tensor_len(tensor_ptr: u64) -> u32;
    pub fn ssvm_tensorflow_get_tensor_data(tensor_ptr: u64, buf: *mut u8);
    pub fn ssvm_tensorflow_append_input(
        input_name: *const u8,
        input_name_len: u32,
        index: u32,
        tensor_ptr: u64,
    );
    pub fn ssvm_tensorflow_append_output(output_name: *const u8, output_name_len: u32, index: u32);
    pub fn ssvm_tensorflow_clear_input();
    pub fn ssvm_tensorflow_clear_output();
    pub fn ssvm_tensorflow_get_result_len() -> u32;
    pub fn ssvm_tensorflow_get_result(buf: *mut u8);
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

/// The session arguments structure.
pub struct SessionArgs {
    input_names: Vec<String>,
    input_idx: Vec<u32>,
    input_tensor: Vec<u64>,
    output_names: Vec<String>,
    output_idx: Vec<u32>,
}

impl SessionArgs {
    pub fn new() -> SessionArgs {
        SessionArgs {
            input_names: vec![],
            input_idx: vec![],
            input_tensor: vec![],
            output_names: vec![],
            output_idx: vec![],
        }
    }

    pub fn add_input<T: TensorType>(&mut self, name: &str, tensor_buf: &[T], shape: &[i64]) {
        // Append input name and operation index.
        let name_pair: Vec<&str> = name.split(":").collect();
        self.input_names.push(String::from(name_pair[0]));
        let idx = if name_pair.len() > 1 {
            name_pair[1].parse().unwrap()
        } else {
            0
        };
        self.input_idx.push(idx);

        // Create input tensor.
        unsafe {
            self.input_tensor.push(ssvm_tensorflow_alloc_tensor(
                shape.as_ptr() as *const u8,
                shape.len() as u32,
                T::val(),
                tensor_buf.as_ptr() as *const u8,
                (tensor_buf.len() * mem::size_of::<T>()) as u32,
            ))
        }
    }

    pub fn add_output(&mut self, name: &str) {
        // Append output name and operation index.
        let name_pair: Vec<&str> = name.split(":").collect();
        self.output_names.push(String::from(name_pair[0]));
        let idx = if name_pair.len() > 1 {
            name_pair[1].parse().unwrap()
        } else {
            0
        };
        self.output_idx.push(idx);
    }
}

impl Drop for SessionArgs {
    fn drop(&mut self) {
        for &ptr in &self.input_tensor {
            unsafe {
                ssvm_tensorflow_delete_tensor(ptr);
            }
        }
    }
}

/// The output tensor structure.
pub struct Tensors {
    names: Vec<String>,
    idx: Vec<u32>,
    tensor: Vec<u64>,
}

impl Tensors {
    pub fn get_output<T: TensorType>(&self, name: &str) -> Vec<T> {
        let name_pair: Vec<&str> = name.split(":").collect();
        let get_idx = if name_pair.len() > 1 {
            name_pair[1].parse().unwrap()
        } else {
            0
        };

        for i in 0..self.names.len() {
            if self.names[i] == name_pair[0] && self.idx[i] == get_idx {
                unsafe {
                    let buf_len = ssvm_tensorflow_get_tensor_len(self.tensor[i]) as usize;
                    let mut data: Vec<T> =
                        vec![T::zero(); buf_len / mem::size_of::<T::InnerType>()];
                    ssvm_tensorflow_get_tensor_data(self.tensor[i], data.as_mut_ptr() as *mut u8);
                    return data;
                }
            }
        }
        return Vec::new();
    }
}

impl Drop for Tensors {
    fn drop(&mut self) {
        for &ptr in &self.tensor {
            unsafe {
                ssvm_tensorflow_delete_tensor(ptr);
            }
        }
    }
}

/// Convert JPEG image in memory into rgb u8 vector.
pub fn load_jpg_image_to_rgb8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    unsafe {
        ssvm_tensorflow_load_jpg_to_rgb8(img_buf.as_ptr() as *const u8, img_buf.len() as u32, w, h);
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<u8> = vec![0; result_len as usize];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr);
        result_vec
    }
}

/// Convert JPEG image in memory into bgr u8 vector.
pub fn load_jpg_image_to_bgr8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    unsafe {
        ssvm_tensorflow_load_jpg_to_bgr8(img_buf.as_ptr() as *const u8, img_buf.len() as u32, w, h);
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<u8> = vec![0; result_len as usize];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr);
        result_vec
    }
}

/// Convert JPEG image in memory into rgb f32 vector.
pub fn load_jpg_image_to_rgb32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    unsafe {
        ssvm_tensorflow_load_jpg_to_rgb32f(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
        );
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<f32> = vec![0.0; result_len as usize / mem::size_of::<f32>()];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr as *mut u8);
        result_vec
    }
}

/// Convert JPEG image in memory into bgr f32 vector.
pub fn load_jpg_image_to_bgr32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    unsafe {
        ssvm_tensorflow_load_jpg_to_bgr32f(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
        );
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<f32> = vec![0.0; result_len as usize / mem::size_of::<f32>()];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr as *mut u8);
        result_vec
    }
}

/// Convert PNG image in memory into rgb u8 vector.
pub fn load_png_image_to_rgb8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    unsafe {
        ssvm_tensorflow_load_png_to_rgb8(img_buf.as_ptr() as *const u8, img_buf.len() as u32, w, h);
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<u8> = vec![0; result_len as usize];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr);
        result_vec
    }
}

/// Convert PNG image in memory into bgr u8 vector.
pub fn load_png_image_to_bgr8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    unsafe {
        ssvm_tensorflow_load_png_to_bgr8(img_buf.as_ptr() as *const u8, img_buf.len() as u32, w, h);
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<u8> = vec![0; result_len as usize];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr);
        result_vec
    }
}

/// Convert PNG image in memory into rgb f32 vector.
pub fn load_png_image_to_rgb32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    unsafe {
        ssvm_tensorflow_load_png_to_rgb32f(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
        );
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<f32> = vec![0.0; result_len as usize / mem::size_of::<f32>()];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr as *mut u8);
        result_vec
    }
}

/// Convert PNG image in memory into bgr f32 vector.
pub fn load_png_image_to_bgr32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    unsafe {
        ssvm_tensorflow_load_png_to_bgr32f(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
        );
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<f32> = vec![0.0; result_len as usize / mem::size_of::<f32>()];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr as *mut u8);
        result_vec
    }
}

/// Run tensorflow model with image input tensor.
pub fn exec_model(model_buf: &[u8], args: &SessionArgs) -> Tensors {
    unsafe {
        for i in 0..args.input_names.len() {
            let input_name_cstr = CString::new(args.input_names[i].to_string()).expect("");
            ssvm_tensorflow_append_input(
                input_name_cstr.as_ptr() as *const u8,
                input_name_cstr.as_bytes().len() as u32,
                args.input_idx[i],
                args.input_tensor[i],
            );
        }

        for i in 0..args.output_names.len() {
            let output_name_cstr = CString::new(args.output_names[i].to_string()).expect("");
            ssvm_tensorflow_append_output(
                output_name_cstr.as_ptr() as *const u8,
                output_name_cstr.as_bytes().len() as u32,
                args.output_idx[i],
            );
        }

        let mut out_tensors: Vec<u64> = vec![0; args.output_names.len()];
        ssvm_tensorflow_exec_model(
            model_buf.as_ptr() as *const u8,
            model_buf.len() as u32,
            out_tensors.as_mut_ptr() as *mut u8,
        );

        ssvm_tensorflow_clear_input();
        ssvm_tensorflow_clear_output();

        Tensors {
            names: args.output_names.clone(),
            idx: args.output_idx.clone(),
            tensor: out_tensors,
        }
    }
}
