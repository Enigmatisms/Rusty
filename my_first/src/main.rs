#[derive(Debug)]

struct TestStruct {
    pub name: String,
    age: i32,
    gender: bool
}

enum Gender {
    Man {id:i32, tag:i32},
    Woman {id:i32, intend:i32},
}

fn main() {
    use std::fs::File;

    let f = File::open("hello.txt");

    let person = TestStruct{
        name: String::from("Bob"),
        age: 18,
        gender: true
    };

    println!("{}, {}, {}", person.name, person.age, person.gender);
    println!("{:#?}", person);
    
    match f {
        Ok(ref file) => {
            println!("File opened successfully. {:#?}", file);
        },
        Err(ref err) => {
            println!("Failed to open the file, {}", err);
        }
    }
    // tryPrint(a);            // value used after move，此处的a已经无效了
}

fn unknowGender(person: &Gender) {
    match person {
        Gender::Man {id, tag} => {
            println!("This is a man: {}, {}", id, tag);
        },
        Gender::Woman {id, intend} => {
            println!("Hello my lady: {}, {}", id, intend);
        }
    }
}

fn testFor() {
    let mut a = [1, 2, 3, 4, 5, 6];
    for i in 0..a.len() {
        println!("a[{}] is {}", i, a[i]);
        a[i] += 1;
    }
    for i in (0..a.len()).rev() {
        println!("a[{}] is {}", i, a[i]);
    }

    // 损失精度转换是不可以的，float64（x）- 3（int）也是不可以的
    // let x:i32 = (x - 3.).into();
    // println!("x is now {}", x);
}

fn tryPrint(string: String) {
    println!("String printed: {}", string);
}

// 实际上是一个表达式块？加上分号成了语句就是错的了
fn noReturn(string: String) -> String {
    let output:String = String::from("Test"); 
    output
}

