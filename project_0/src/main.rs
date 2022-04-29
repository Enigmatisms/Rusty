use rand::Rng;
use std::time::{Duration, Instant};

fn main() {
    sort_test(& quick_sort);
}

fn sort_test(sort_func: &dyn Fn(&mut [i32])) {
    let max_num:usize = 1000;
    let mut rng = rand::thread_rng();
    let mut arr = [0; 1000];
    for i in 0..max_num {
        arr[i] = rng.gen_range(0..(max_num as i32));
    }
    print_arr(& arr);
    let start = Instant::now();
    sort_func(&mut arr);
    let duration = start.elapsed();
    println!("\nAfter sort:");
    print_arr(& arr);
    println!("Time elapsed: {:?}", duration);
}

fn print_arr(arr: & [i32]) {
    for num in arr {
        print!("{}, ", num);
    }
    print!("\n");
}

fn divide_sort_func(arr: &[i32]) -> Vec<i32> {
    let now_len = arr.len();
    if now_len > 2 {
        let half_len = now_len >> 1;
        let back_len = now_len - half_len;
        let front_half: Vec<i32> = divide_sort_func(&arr[..half_len]);
        let rare_half: Vec<i32> = divide_sort_func(&arr[half_len..]);
        let mut front_p = 0;
        let mut rare_p = 0;
        let mut result: Vec<i32> = Vec::new(); 
        // 是否有更简洁的写法？
        while front_p < half_len && rare_p < back_len {
            if front_half[front_p] <= rare_half[rare_p] {
                result.push(front_half[front_p]);
                front_p += 1;
            } else {
                result.push(rare_half[rare_p]);
                rare_p += 1;
            }
        }
        for i in front_p..half_len {
            result.push(front_half[i]);
        }
        for i in rare_p..back_len {
            result.push(rare_half[i]);
        }
        return result;
    } else if now_len == 2{
        if arr[0] > arr[1] {
            return vec![arr[1], arr[0]];
        }
        return vec![arr[0], arr[1]];
    } else {
        return vec![arr[0]];
    }
}

fn merge_sort(arr: &mut [i32]){
    let result_vec: Vec<i32> = divide_sort_func(arr);
    for i in 0..arr.len() {
        arr[i] = result_vec[i];
    }
}

fn quick_sort_func(arr: &mut [i32]) {
    let length = arr.len();
    println!("{}", length);
    if length > 2 {
        let mut begin_is_pivot: bool = false;
        let mut begin = 0;
        let mut end = length - 1;
        let pivot = arr[begin];
        while arr[begin] == pivot && arr[end] == pivot && begin < end {
            begin += 1;
            end -= 1;
        }
        if arr[begin] == pivot {
            begin_is_pivot = true;
        }
        while begin < end {
            if arr[begin] <= arr[end]{
                if begin_is_pivot {
                    end -= 1;
                } else {
                    begin += 1;
                }
            } else {
                let tmp = arr[begin];
                arr[begin] = arr[end];
                arr[end] = tmp;
                begin_is_pivot = !begin_is_pivot;
            }
        }
        if begin == length || begin == 0 {
            return;
        }
        quick_sort_func(&mut arr[0..begin]);
        quick_sort_func(&mut arr[begin..length]);
    } else if length == 2 {
        if arr[0] > arr[1] {
            let tmp = arr[0];
            arr[0] = arr[1];
            arr[1] = tmp;
        }
    }
}

fn quick_sort(arr: &mut [i32]) {
    quick_sort_func(arr);
}
