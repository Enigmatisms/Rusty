// 双向链表
use std::rc::Rc;
use std::cell::{RefCell, Ref, RefMut};
use std::fmt::Display;

pub struct DoubleList<T> {
    head: Link<T>,
    tail: Link<T>
}

type Link<T> = Option<Rc<RefCell<Node<T>>>>;

struct Node<T> {
    elem: T,
    prev: Link<T>,
    next: Link<T>
}

impl<T> DoubleList<T> {
    pub fn new() -> DoubleList<T> {
        DoubleList{
            head: None,
            tail: None
        }
    }
}

impl<T> DoubleList<T> {
    pub fn push_front(&mut self, elem: T) {
        // 新的Node需要设置head与tail
        let new_node = Node {
            elem: elem,
            prev: None,
            next: self.head.clone()
        };
        let wrapped_node = Rc::new(RefCell::new(new_node));
        if let Some(node) = self.head.take() {
            let mut inner_node = node.borrow_mut();
            // let _:() = inner_node;
            inner_node.prev = Some(wrapped_node.clone());           // Rc的clone应该开销比较小
            self.head = Some(wrapped_node);
        } else {                                        // 如果原来链表为空，则需要设置tail
            self.tail = Some(wrapped_node.clone());
            self.head = Some(wrapped_node);
        }
    }

    pub fn push_back(&mut self, elem: T) {
        let new_node = Node {
            elem: elem,
            prev: self.tail.clone(),
            next: None
        };
        let wrapped_node = Some(Rc::new(RefCell::new(new_node)));
        if let Some(node) = self.tail.take() {
            let mut inner_node = node.borrow_mut();
            // let _:() = inner_node;
            inner_node.next = wrapped_node.clone();           // Rc的clone应该开销比较小
            self.tail = wrapped_node;
        } else {                                        // 如果原来链表为空，则需要设置tail
            self.tail = wrapped_node.clone();
            self.head = wrapped_node;
        }
    }
    

    pub fn pop_front(&mut self) -> Option<T> {
        self.head.take().map(|node|{            // node 类型是 Rc<RefCell<Node<T>>>
            if let Some(next_node) = node.borrow_mut().next.take() {
                next_node.borrow_mut().prev.take();         // 消除一个ref_count
                self.head = Some(next_node);
            } else {                                        // 只有head（tail）
                self.tail.take();
            }
            Rc::try_unwrap(node).ok().unwrap().into_inner().elem
        })
    }

    pub fn pop_back(&mut self) -> Option<T> {
        self.tail.take().map(|node|{
            if let Some(prev_node) = node.borrow_mut().prev.take() {
                prev_node.borrow_mut().next.take();
                self.tail = Some(prev_node);
            } else {
                self.head.take();
            }
            Rc::try_unwrap(node).ok().unwrap().into_inner().elem
        })
    }

    pub fn peek_front(& self) -> Option<Ref<T>> {
        self.head.as_ref().map(|node| {
            Ref::map(node.borrow(), |inner_node| {&inner_node.elem})
        })  
    }

    pub fn peek_back(& self) -> Option<Ref<T>> {
        self.tail.as_ref().map(|node| {
            Ref::map(node.borrow(), |inner_node| {&inner_node.elem})
        }) 
    }

    pub fn peek_front_mut(&mut self) -> Option<RefMut<T>> {
        self.head.as_mut().map(|node| {
            RefMut::map(node.borrow_mut(), |inner_node| {&mut inner_node.elem})
        }) 
    }

    pub fn peek_back_mut(&mut self) -> Option<RefMut<T>> {
        self.tail.as_mut().map(|node| {
            RefMut::map(node.borrow_mut(), |inner_node| {&mut inner_node.elem})
        }) 
    }
}

impl<T: Display> DoubleList<T> {
    pub fn show_links(&self) {
        let mut ptr = self.head.clone();
        while let Some(rc) = ptr {
            ptr = rc.borrow().next.clone();
            print!("{}, ", rc.borrow().elem);
        }
        print!("\n");
    }
}

impl<T> Drop for DoubleList<T> {
    // 显然还有更简单的方法，直接复用pop
    fn drop(&mut self) {
        let mut ptr = self.head.take();
        while let Some(node) = ptr {
            node.borrow_mut().prev.take();
            ptr = node.borrow_mut().next.take();
        }
        self.tail.take();
    }
}

// peeking的实现，显然，由于pop方法已经实现了，into_iter是非常好实现的
// 类比C++，此结构可以有reverse iterator，但是实现应该是差不多的，故不做要求
pub struct IntoIterBase<T, const FLAG: bool> {
    this: DoubleList<T>
}
type IntoIter<T> = IntoIterBase<T, true>;
type IntoIterRev<T> = IntoIterBase<T, false>;
impl<T> DoubleList<T> {
    pub fn into_iter<const FLAG: bool>(self) -> IntoIterBase<T, FLAG> {
        IntoIterBase {
            this: self
        }
    }
}

impl<T, const FLAG: bool> Iterator for IntoIterBase<T, FLAG> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if FLAG == true {
            if let Some(result) = self.this.pop_front() {
                return Some(result);
            }
        } else {
            if let Some(result) = self.this.pop_back() {
                return Some(result);
            }
        }
        None
    }
}

// =========== 返回Ref可以实现Iter吗 ============
pub struct IterBase<'a, T, const FLAG: bool> {
    this: Option<Ref<'a, Node<T>>>
}
type Iter<'a, T> = IterBase<'a, T, true>;
type IterRev<'a, T> = IterBase<'a, T, false>;
impl<T> DoubleList<T> {
    pub fn iter(&self) -> Iter<T> {
        IterBase {
            this: self.head.as_ref().map(|rc_node|{
                rc_node.borrow()
            })
        }
    }
    pub fn reverse_iter(&self) -> IterRev<T> {
        IterRev {
            this: self.tail.as_ref().map(|rc_node|{
                rc_node.borrow()
            })
        }
    }
}

// 这个我还是实现不了
impl<'a, T, const FLAG: bool> Iterator for IterBase<'a, T, FLAG> {
    type Item = Ref<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        if FLAG == true {
            if let Some(ref_val) = self.this.as_ref() {       // ref_val 是 Ref<'a, Node<T>> *之后成为Node<T>
                None
            } else {
                None
            }
        } else {
            None
        }
    }
}

mod test{
    #[test]
    fn pop_back_test() {
        let mut dl = super::DoubleList::new();
        dl.push_back(4);
        dl.push_back(5);
        dl.push_back(6);
        dl.push_front(1);
        dl.push_front(2);
        assert_eq!(dl.pop_back(), Some(6));
        assert_eq!(dl.pop_back(), Some(5));
        assert_eq!(dl.pop_back(), Some(4));
        assert_eq!(dl.pop_back(), Some(1));
        assert_eq!(dl.pop_back(), Some(2));
        assert_eq!(dl.pop_back(), None);
    }

    #[test]
    fn pop_front_test() {
        let mut dl = super::DoubleList::new();
        dl.push_front(1);
        dl.push_front(2);
        dl.push_back(4);
        dl.push_back(5);
        dl.push_back(6);
        assert_eq!(dl.pop_front(), Some(2));
        assert_eq!(dl.pop_front(), Some(1));
        assert_eq!(dl.pop_front(), Some(4));
        assert_eq!(dl.pop_front(), Some(5));
        assert_eq!(dl.pop_front(), Some(6));
        assert_eq!(dl.pop_back(), None);
    }

    #[test]
    fn casual_test() {
        let mut dl = super::DoubleList::new();
        dl.push_front(1);
        dl.push_back(4);
        assert_eq!(dl.pop_front(), Some(1));
        assert_eq!(dl.pop_front(), Some(4));
        dl.push_back(5);
        dl.push_front(6);
        dl.push_front(7);
        dl.push_front(8);
        assert_eq!(dl.pop_front(), Some(8));
        assert_eq!(dl.pop_back(), Some(5));
        assert_eq!(dl.pop_front(), Some(7));
        assert_eq!(dl.pop_back(), Some(6));
        assert_eq!(dl.pop_front(), None);
        assert_eq!(dl.pop_back(), None);
    }

    #[test]
    fn peek_test() {
        let mut dl = super::DoubleList::new();
        dl.push_front(1);
        dl.push_front(2);
        dl.push_back(4);
        assert_eq!(*dl.peek_front().unwrap(), 2);
        assert_eq!(*dl.peek_back().unwrap(), 4);
        dl.push_back(3);
        assert_eq!(*dl.peek_back().unwrap(), 3);
        dl.pop_front();
        assert_eq!(*dl.peek_front().unwrap(), 1);
    }

    #[test]
    fn peek_mut_test() {
        let mut dl = super::DoubleList::new();
        dl.push_front(1);
        dl.push_front(2);
        dl.push_back(4);
        let top = dl.peek_front_mut();
        *(top.unwrap()) = 5;
        assert_eq!(*dl.peek_front().unwrap(), 5);
        assert_eq!(dl.pop_front(), Some(5));

        let top = dl.peek_front_mut();
        assert_eq!(*(top.unwrap()), 1);

        let bottom = dl.peek_back_mut();
        *(bottom.unwrap()) = -1;
        assert_eq!(dl.pop_back(), Some(-1));
    }

    #[test]
    fn iter_test() {
        let mut dl = super::DoubleList::new();
        dl.push_front(1);
        dl.push_front(2);
        dl.push_front(6);
        dl.push_back(3);
        let mut it: super::IntoIter<i32> = dl.into_iter::<true>();
        assert_eq!(it.next(), Some(6));
        assert_eq!(it.next(), Some(2));
        assert_eq!(it.next(), Some(1));
        assert_eq!(it.next(), Some(3));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn rev_iter_test() {
        let mut dl = super::DoubleList::new();
        dl.push_front(1);
        dl.push_front(2);
        dl.push_front(6);
        dl.push_back(3);
        let mut it: super::IntoIterRev<i32> = dl.into_iter::<false>();
        assert_eq!(it.next(), Some(3));
        assert_eq!(it.next(), Some(1));
        assert_eq!(it.next(), Some(2));
        assert_eq!(it.next(), Some(6));
        assert_eq!(it.next(), None);
    }
}
