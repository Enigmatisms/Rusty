mod first;
use rand::Rng;
use std::time::Instant;

fn link_test() {
    let mut l1: first::List<i32> = first::List::new(); 
    let mut l2: first::List<i32> = first::List::new(); 
    let max_num:usize = 2000;
    let mut rng = rand::thread_rng();
    let mut arr = [0; 2000];
    for i in 0..max_num {
        arr[i] = rng.gen_range(0..(max_num as i32));
    }
    println!("Test sequence started.");
    let start = Instant::now();
    for elem in arr {
        l1.clone_push(elem);
    }
    let duration = start.elapsed();
    println!("Time elapsed for clone push: {:?}", duration);
    let start = Instant::now();
    for elem in arr {
        l2.push(elem);
    }
    let duration = start.elapsed();
    println!("Time elapsed for replace push: {:?}", duration);
}

fn test_push_pop() {
    let mut l1: first::List<i32> = first::List::new(); 
    l1.push(3);
    l1.push(2);
    l1.push(4);
    l1.push(6);
    l1.push(1);
    l1.show_stack(true, true);
    for i in 0..6 {
        println!("Stack popped: {}", l1.pop());
        l1.show_stack(true, true);
    }
}

fn main() {
    test_push_pop();
}