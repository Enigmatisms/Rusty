// 双向链表的实现，借助此双向链表以及基于此链表实现的双端队列巩固Rust基本操作


pub struct DeList<T> {
    head: Link<T>,
    tail: Link<T>,
    size: u32
}

type Link<T> = Option<Box<Node<T>>>;

struct Node<T> {
    elem: T,
    prev: Link<T>,
    next: Link<T>
}

impl<T> DeList<T> {
    pub fn push_front(&mut self, new_elem: T) {

    }

    pub fn push_back(&mut self, new_elem: T) {

    }

    pub fn pop_front(&mut self) -> Option<T> {

    }

    pub fn pop_back(&mut self) -> Option<T> {

    }

    pub fn len(& self) -> u32 {

    }

    pub fn front() {

    }

    pub fn back() {

    }

    pub fn front_mut() {

    }

    pub fn back_mut() {

    }
}

impl<T: Display> DeList<T> {
    pub fn show_delist(reverse: bool) {
        ;
    }
} 

// Only types with trait Copy can use this implement (which will not destroy the original variables)
impl<T: Copy> DeList<T> {
    pub fn copy_push_front(&mut self, new_elem: &T) {

    }

    pub fn copy_push_back(&mut self, new_elem: &T) {

    }
}

// Rust compile time polymorphism
struct IntoIterBase<T, const REV: bool>(List<T>);
type IntoIter<T> = IntoIterBase<T, true>;
type IntoIterRev<T> = IntoIterBase<T, false>;
impl<T> DeList<T> {
    pub fn into_iter(self) -> IntoIter<T> {
        IntoIter(self.head)
    }

    pub fn into_iter_reverse(self) -> IntoIterRev<T> {
        IntoIterRev(self.tail)
    }
}
impl<T, const REV: bool> Iterator for IntoIterBase<T, REV> {
    type Item = T;
    fn next(self) -> Option<Self::Item> {
        if REV == true {

        } else {

        }
    }
}

// 不可变引用与可变引用迭代器都需要实现
