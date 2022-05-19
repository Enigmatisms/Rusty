extern crate libc;

#[link(name = "cuda_add", kind = "static")]

#[repr(C)]
pub struct Vec4 {
    x: libc::c_float,
    y: libc::c_float,
    z: libc::c_float,
    num: libc::c_int
}

extern {
    // fn test_array_increment(ioput: *mut libc::c_float, input: *const libc::c_float);
    fn cuda_add_array(input: *const libc::c_float, output: *mut libc::c_float, total_num: Vec4);
}

// fn test_c() {
//     let mut mut_array: [f32; 3600] = [0.; 3600];
//     let im_array: [f32; 3600] = [2.; 3600];
//     unsafe {test_array_increment(mut_array.as_mut_ptr(), im_array.as_ptr());}
//     println!("Array after mut:");
//     for i in 0..3600 {
//         print!("{}, ", mut_array[i]);
//     }
// }

fn main() {
    let mut array1: [f32; 32768] = [0.; 32768];
    let mut array2: [f32; 32768] = [0.; 32768];
    for i in 0..32768 {
        array1[i] = i as f32;
        array2[i] = -(i as f32);
    }
    let vec: Vec4 = Vec4 { x: 2.3, y: -1.5, z: 5.0, num: 32768 };
    unsafe {cuda_add_array(array2.as_ptr(), array1.as_mut_ptr(), vec);}
    println!("Array after cuda addition:");
    for i in 0..128 {
        print!("{}, ", array1[i]);
    }
    print!("\n");
}