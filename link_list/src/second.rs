use std::mem;
use std::fmt::Display;

pub struct List<T> where T: Default {
    head: Link<T>,
}

type Link<T> = Option<Box<Node<T>>>;

struct Node<T> where T: Default {
    elem: T,
    next: Link<T>,
}

impl<T: Default> List<T> {
    pub fn new() -> Self {
        List { head: None }
    }
}

impl<T: Default> List<T> {
    pub fn push(&mut self, elem: T) {
        let new_node: Node<T> = Node {
            elem: elem,
            next: self.head.take()
        };
        self.head = Some(Box::new(new_node));
    }

    pub fn pop(&mut self) -> T {
        self.head.take().map(|node| {
            self.head = node.next;
            node.elem
        }).unwrap()
    }

    pub fn top(&self) -> &T {
        self.head.as_ref().map(|node|{
            &node.elem
        }).unwrap()
    }
}

impl<T: std::fmt::Display + Default> List<T> {
    pub fn show_stack(&self, verbose:bool, single_line:bool) {
        if verbose {
            println!("Current stack:");
        }
        let mut ptr: &Link<T> = &self.head;
        while let Some(node) = ptr {
            if single_line {
                print!("{}, ", node.elem);
            } else {
                println!("Stack: {}", node.elem);
            }
            ptr = &node.next;
        }
        if single_line {
            println!("");
        }
    }
}

impl<T: Default> Drop for List<T> {
    fn drop(&mut self) {
        let mut ptr = self.head.take();
        while let Some(mut node) = ptr{
            ptr = node.next.take();
        }
    }
}

mod test {
    #[test]
    fn basics() {
        let mut l1: super::List<i32> = super::List::new(); 
        
        l1.push(4);
        l1.push(6);
        l1.push(1);
        assert_eq!(l1.pop(), 1);
        assert_eq!(l1.pop(), 6);
        assert_eq!(l1.pop(), 4);
        l1.push(3);
        l1.push(2);
        l1.push(5);
        assert_eq!(l1.pop(), 5);
        assert_eq!(l1.pop(), 2);
        l1.push(8);
        assert_eq!(l1.pop(), 8);
        assert_eq!(l1.pop(), 3);
    }
}