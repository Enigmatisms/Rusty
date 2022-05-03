use std::mem;
use std::fmt::Display;

pub struct List<T> {
    head: Link<T>,
}

type Link<T> = Option<Box<Node<T>>>;

struct Node<T> {
    elem: T,
    next: Link<T>,
}

impl<T> List<T> {
    pub fn new() -> Self {
        List { head: None }
    }
}

impl<T> List<T> {
    pub fn push(&mut self, elem: T) {
        let new_node: Node<T> = Node {
            elem: elem,
            next: self.head.take()
        };
        self.head = Some(Box::new(new_node));
    }

    pub fn pop(&mut self) -> Option<T> {
        self.head.take().map(|node| {
            self.head = node.next;
            node.elem
        })
    }

    pub fn pop_unwrap(&mut self) -> T {
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

    // the type of the returned value should be mut
    pub fn mut_top(&mut self) -> &mut T {
        self.head.as_mut().map(|node|{
            &mut node.elem
        }).unwrap()
    }
}

pub struct IntoIter<T>(List<T>) ;
impl<T> List<T> {
    pub fn into_iter(self) -> IntoIter<T> {
        IntoIter(self)
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        // access fields of a tuple struct numerically
        self.0.pop()
    }
}

// =================================================
pub struct Iter<'a, T> where T: Default{			// 这里应该表示的是，Iter能存活多久，this指向的内容就应该存活多久（因为this引用的lifetime现在与Iter一致）
    this: &'a Link<T>
}
impl<T: Default> List<T> {				 // List本身不需要显式lifetime概念，故不用写
    pub fn iter(&self) -> Iter<T> {			// 根据lifetime省略第三规则，输入有&self，则输出的lifetime默认与self一致，这样iter的lifetime会与链表本身一致，内部的引用也一样
        Iter{this: &self.head}
    }
} 
impl<'a, T> Iterator for Iter<'a, T> where T: Default {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.this.as_ref().map(|node|{
            self.this = &node.next;
            &node.elem
        })
    }
}
// ============= 以下是官方教程的实现 ===============
// pub struct Iter<'a, T> {
//     next: Option<&'a Box<Node<T>>>,
// }

// impl<T> List<T> {
//     pub fn iter<'a>(&'a self) -> Iter<'a, T> {
//         Iter { next: self.head.as_ref().map(|node| node) }
//     }
// }

// impl<'a, T> Iterator for Iter<'a, T> {
//     type Item = &'a T;

//     fn next(&mut self) -> Option<Self::Item> {
//         self.next.map(|node| {
//             self.next = node.next.as_ref().map(|node| node);
//             &node.elem
//         })
//     }
// }
// ==================================================

pub struct IterMut<'a, T> where T: Default {
    this: Option<&'a mut Node<T>>
}

impl<T: Default> List<T> {
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            this: self.head.as_deref_mut()
        }
    }
}

impl<'a, T> Iterator for IterMut<'a, T> where T: Default {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.this.take() {
            self.this = node.next.as_deref_mut();
            Some(&mut node.elem)
        } else {
            None
        }
        // self.this.as_mut().map(|node|{
        //     // self.this = &mut node.next;
        //     &mut node.elem
        // })
    }
}

impl<T: std::fmt::Display> List<T> {
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

impl<T> Drop for List<T> {
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
        assert_eq!(l1.pop_unwrap(), 1);
        assert_eq!(l1.pop_unwrap(), 6);
        assert_eq!(l1.pop_unwrap(), 4);
        l1.push(3);
        l1.push(2);
        l1.push(5);
        assert_eq!(l1.pop_unwrap(), 5);
        assert_eq!(l1.pop_unwrap(), 2);
        l1.push(8);
        assert_eq!(l1.pop_unwrap(), 8);
        assert_eq!(l1.pop_unwrap(), 3);
        l1.push(7);
        l1.push(9);
        assert_eq!(*l1.top(), 9);
        l1.pop_unwrap();
        assert_eq!(*l1.top(), 7);
        let top_ref = l1.mut_top();
        *top_ref = 1;
        assert_eq!(*l1.top(), 1);
    }
    #[test]
    fn into_iter_test() {
        let mut l1: super::List<i32> = super::List::new(); 
        l1.push(4);
        l1.push(6);
        l1.push(1);
        let mut iterator = l1.into_iter();
        assert_eq!(iterator.next(), Some(1));
        assert_eq!(iterator.next(), Some(6));
        assert_eq!(iterator.next(), Some(4));
        assert_eq!(iterator.next(), None);
    }
    // #[test]
    // fn iter_test() {
    //     let mut l1: super::List<i32> = super::List::new(); 
    //     l1.push(4);
    //     l1.push(6);
    //     l1.push(1);
    //     let mut iterator = l1.iter();
    //     assert_eq!(iterator.next(), Some(&1));
    //     assert_eq!(iterator.next(), Some(&6));
    //     assert_eq!(iterator.next(), Some(&4));
    //     assert_eq!(iterator.next(), None);
    // }
}