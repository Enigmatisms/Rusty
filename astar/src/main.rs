mod quick_test;

fn test_heap() {
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;
    let t1 = quick_test::Test {
        score: 5,
        name: String::from("Bob"),
        id: 218
    };
    let t2 = quick_test::Test {
        score: 7,
        name: String::from("Alice"),
        id: 217
    };
    let t3 = quick_test::Test {
        score: 3,
        name: String::from("Steve"),
        id: 666
    };
    let t4 = quick_test::Test {
        score: 2,
        name: String::from("Jeff"),
        id: 0
    };
    let t5 = quick_test::Test {
        score: 4,
        name: String::from("Lex"),
        id: 1255
    };
    let mut heap: BinaryHeap<Reverse<quick_test::Test>> = BinaryHeap::new();
    heap.push(Reverse(t1));
    heap.push(Reverse(t2));
    heap.push(Reverse(t3));
    heap.push(Reverse(t4));
    heap.push(Reverse(t5));
    while let Some(popped) = heap.pop() {
        println!("name:{}, score:{}, id:{}", &popped.0.name, &popped.0.score, &popped.0.id);
    }
}

fn test_hash () {
    use std::collections::HashMap;
    let mut hash_map: HashMap<i32, quick_test::Pos3> = HashMap::new();
    let p1 = quick_test::Pos3(6, 9, 23);
    let p2 = quick_test::Pos3(5, 3, 120);
    let p3 = quick_test::Pos3(9, 17, 332);
    let p4 = quick_test::Pos3(72, 4, 15);
    let p5 = quick_test::Pos3(0, 6, 0);
    let tp1 = quick_test::Pos3(6, 9, 89);
    let tp2 = quick_test::Pos3(4, 8, 89);
    hash_map.insert(p1.hash(), p1);
    hash_map.insert(p2.hash(), p2);
    hash_map.insert(p3.hash(), p3);
    hash_map.insert(p4.hash(), p4);
    hash_map.insert(p5.hash(), p5);
    println!("tp1 ({}, {}, {}) in hash_map ? {}", tp1.0, tp1.1, tp1.2, hash_map.contains_key(&tp1.hash()));
    println!("tp2 ({}, {}, {}) in hash_map ? {}", tp2.0, tp2.1, tp2.2, hash_map.contains_key(&tp2.hash()));
}

fn test_array2d() {
    use array2d::Array2d;
    let rows = vec![
        vec![1, 1, 1, 1, 1, 1, 1],
        vec![1, 0, 0, 0, 0, 0, 1],
        vec![1, 1, 1, 1, 1, 0, 1],
        vec![1, 0, 0, 0, 1, 0, 1],
        vec![1, 0, 1, 0, 1, 0, 1],
        vec![1, 0, 1, 0, 0, 0, 1],
        vec![1, 1, 1, 1, 1, 1, 1],
    ];
    let array = Array2D::from_rows(&rows);
    println!("arr[{}, {}] = {}", 3, 4, array[(3, 4)]);
    println!("arr[{}, {}] = {}", 0, 0, array[(0, 0)]);
    println!("arr[{}, {}] = {}", 6, 6, array[(6, 6)]);
    println!("arr[{}, {}] = {}", 5, 5, array[(5, 5)]);
}

fn main() {
    test_array2d();
}