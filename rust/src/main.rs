use std::{fmt::Debug, process::Command, task::Context};

const KEY_CHARS: [u8; 63] = [
    b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o', b'p',
    b'q', b'r', b's', b't', b'u', b'v', b'w', b'x', b'y', b'z', b'A', b'B', b'C', b'D', b'E', b'F',
    b'G', b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O', b'P', b'Q', b'R', b'S', b'T', b'U', b'V',
    b'W', b'X', b'Y', b'Z', b'_', b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9',
];

#[derive(Debug, Eq, PartialEq)]
pub struct OrderedMap<K: Eq, T>(Vec<(K, T)>);

impl<K: Eq + Debug, T: Debug> OrderedMap<K, T> {
    pub fn new(items: Vec<(K, T)>) -> Self {
        let mut keys: Vec<&K> = vec![];
        for (key, value) in &items {
            assert!(!keys.contains(&key));
            keys.push(key);
        }
        Self(items)
    }

    fn get(&self, key: &K) -> Option<&T> {
        for (k, v) in &self.0 {
            if key == k {
                return Some(&v);
            }
        }
        None
    }

    fn index(&self, key: &K) -> Option<usize> {
        for i in 0..self.0.len() {
            if &self.0[i].0 == key {
                return Some(i);
            }
        }
        None
    }

    fn key(&self, i: usize) -> &K {
        &self.0[i].0
    }

    fn items(&self) -> &Vec<(K, T)> {
        &self.0
    }

    fn contains(&self, key: &K) -> bool {
        for i in 0..self.0.len() {
            if &self.0[i].0 == key {
                return true;
            }
        }
        false
    }
}

trait Message {
    fn get_type() -> Type
    where
        Self: Sized;
    fn encode(&self) -> Primitive;
    fn decode(v: &Primitive) -> Result<Self, ()>
    where
        Self: Sized;
    fn serialize(&self, ctx: &Lookup) -> Vec<u8>
    where
        Self: Sized,
    {
        Self::get_type()
            .value_to_bytes(ctx, &self.encode())
            .unwrap()
    }
    fn deserialize(ctx: &Lookup, data: &Vec<u8>) -> Self
    where
        Self: Sized,
    {
        let (idx, v) = Self::get_type().value_from_bytes(ctx, 0, data).unwrap();
        assert_eq!(idx, data.len());
        Self::decode(&v).unwrap()
    }
}

#[derive(Eq, PartialEq, Clone)]
pub struct Key(Vec<u8>);

impl Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Key")
            .field(&String::from_utf8(self.0.clone()).unwrap())
            .finish()
    }
}

impl Key {
    pub fn from_string(value: String) -> Self {
        Self::from_bytes(value.chars().map(|x| -> u8 { x as u8 }).collect())
    }

    pub fn from_str(value: &str) -> Self {
        Self::from_bytes(value.chars().map(|x| -> u8 { x as u8 }).collect())
    }

    pub fn from_bytes(value: Vec<u8>) -> Self {
        for c in value.iter() {
            assert!(KEY_CHARS.contains(c));
        }
        Self(value)
    }
}

impl Message for Key {
    fn get_type() -> Type {
        Type::List(Box::new(Type::Byte))
    }

    fn encode(&self) -> Primitive {
        Primitive::Vector(self.0.iter().map(|x| Primitive::Byte(*x)).collect())
    }

    fn decode(v: &Primitive) -> Result<Self, ()> {
        match v {
            Primitive::Vector(bytes_prim) => {
                let mut bytes: Vec<u8> = vec![];

                for byte_prim in bytes_prim {
                    match byte_prim {
                        Primitive::Byte(byte) => bytes.push(*byte),
                        _ => {
                            return Err(());
                        }
                    }
                }

                Ok(Key::from_bytes(bytes))
            }
            _ => Err(()),
        }
    }
}

pub type Lookup = OrderedMap<Key, Type>;

impl Message for Lookup {
    fn get_type() -> Type {
        Type::List(Box::new(Type::Tuple(vec![
            Type::Named(Key::from_str("key")),
            Type::Named(Key::from_str("type")),
        ])))
    }

    fn encode(&self) -> Primitive {
        Primitive::Vector(
            self.items()
                .iter()
                .map(|(k, t)| Primitive::Vector(vec![k.encode(), t.encode()]))
                .collect(),
        )
    }

    fn decode(v: &Primitive) -> Result<Self, ()> {
        match v {
            Primitive::Vector(ws) => {
                let mut options: Vec<(Key, Type)> = vec![];
                for w in ws {
                    match w {
                        Primitive::Vector(pair) => {
                            if let 2 = pair.len() {
                                options.push((
                                    Key::decode(&pair[0]).unwrap(),
                                    Type::decode(&pair[1]).unwrap(),
                                ));
                            } else {
                                return Err(());
                            }
                        }
                        _ => {
                            return Err(());
                        }
                    }
                }
                Ok(Lookup::new(options))
            }
            _ => Err(()),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Primitive {
    Byte(u8),
    Quantity(usize),
    Vector(Vec<Primitive>),
    Varient(Key, Box<Primitive>),
}

#[derive(Debug, Eq, PartialEq)]
pub enum Type {
    Named(Key),
    Byte,
    Quantity,
    Tuple(Vec<Type>),
    List(Box<Type>),
    Varient(Lookup),
}

impl Type {
    fn validate_names(&self, lookup: &Lookup) -> bool {
        match self {
            Type::Tuple(ts) => {
                for t in ts {
                    if (!t.validate_names(lookup)) {
                        return false;
                    }
                }
                true
            }
            Type::List(t) => t.validate_names(lookup),
            Type::Varient(options) => {
                for (k, t) in options.items() {
                    if (!t.validate_names(lookup)) {
                        return false;
                    }
                }
                true
            }
            Type::Named(key) => lookup.contains(&key),
            _ => true,
        }
    }

    fn value_to_bytes(&self, ctx: &Lookup, value: &Primitive) -> Result<Vec<u8>, String> {
        match self {
            Type::Named(key) => match ctx.get(key).unwrap().value_to_bytes(ctx, value) {
                Ok(data) => Ok(data),
                Err(msg) => Err(msg),
            },
            Type::Byte => match value {
                Primitive::Byte(x) => Ok(vec![*x]),
                _ => Err(String::from("Invalid primitive")),
            },
            Type::Quantity => match value {
                Primitive::Quantity(n) => {
                    let mut data: Vec<u8> = vec![];
                    let mut n = *n;
                    while n >= 128 {
                        let r = n % 128;
                        data.push(r as u8 + 128);
                        n = n / 128;
                    }
                    data.push(n as u8);
                    Ok(data)
                }
                _ => Err(String::from("Invalid primitive")),
            },
            Type::Tuple(ts) => match value {
                Primitive::Vector(elems) => {
                    let n = ts.len();
                    if elems.len() != n {
                        return Err(String::from("Tuple has wrong length"));
                    }
                    let mut data: Vec<u8> = vec![];
                    for i in 0..n {
                        match ts[i].value_to_bytes(ctx, &elems[i]) {
                            Ok(elem_data) => data.append(&mut elem_data.clone()),
                            Err(msg) => {
                                return Err(msg);
                            }
                        };
                    }
                    Ok(data)
                }
                _ => Err(String::from("Invalid primitive")),
            },
            Type::List(t) => match value {
                Primitive::Vector(elems) => {
                    let mut data: Vec<u8> = vec![];
                    data.append(
                        &mut Type::Quantity
                            .value_to_bytes(ctx, &Primitive::Quantity(elems.len()))
                            .unwrap(),
                    );
                    for elem in elems {
                        data.append(&mut t.value_to_bytes(ctx, elem).unwrap());
                    }
                    return Ok(data);
                }
                _ => Err(String::from("Invalid primitive")),
            },
            Type::Varient(options) => match value {
                Primitive::Varient(k, v) => {
                    let mut data: Vec<u8> = vec![];

                    //get the index of key k
                    let n = match options.index(k) {
                        Some(n) => n,
                        None => {
                            return Err(format!("Invalid key {:?} for {:?}", k, self));
                        }
                    };

                    data.append(
                        &mut Type::Quantity
                            .value_to_bytes(ctx, &Primitive::Quantity(n))
                            .unwrap(),
                    );

                    data.append(&mut options.get(k).unwrap().value_to_bytes(ctx, v).unwrap());

                    return Ok(data);
                }
                _ => Err(String::from("Invalid primitive")),
            },
        }
    }

    fn value_from_bytes(
        &self,
        ctx: &Lookup,
        idx: usize,
        data: &Vec<u8>,
    ) -> Result<(usize, Primitive), String> {
        match self {
            Type::Named(key) => match ctx.get(key).unwrap().value_from_bytes(ctx, idx, data) {
                Ok((idx, val)) => Ok((idx, val)),
                Err(msg) => Err(msg),
            },
            Type::Byte => Ok((idx + 1, Primitive::Byte(data[idx]))),
            Type::Quantity => {
                let mut idx = idx;
                let mut n: usize = 0;
                let mut p: usize = 0;
                let mut b;
                loop {
                    b = data[idx];
                    idx += 1;
                    n += ((b % 128) as usize) << (p * 7);
                    p += 1;
                    if b / 128 == 0 {
                        break;
                    }
                }
                Ok((idx, Primitive::Quantity(n)))
            }
            Type::Tuple(ts) => {
                let mut idx = idx;
                let mut values: Vec<Primitive> = vec![];
                for t in ts {
                    match t.value_from_bytes(ctx, idx, data) {
                        Ok((new_idx, w)) => {
                            idx = new_idx;
                            values.push(w);
                        }
                        Err(msg) => return Err(msg),
                    }
                }
                Ok((idx, Primitive::Vector(values)))
            }
            Type::List(t) => {
                let mut idx = idx;
                let n_prim;
                (idx, n_prim) = Type::Quantity.value_from_bytes(ctx, idx, data).unwrap();
                match n_prim {
                    Primitive::Quantity(n) => {
                        let mut values: Vec<Primitive> = vec![];
                        for _i in 0..n {
                            let prim;
                            (idx, prim) = t.value_from_bytes(ctx, idx, data).unwrap();
                            values.push(prim);
                        }
                        Ok((idx, Primitive::Vector(values)))
                    }
                    _ => Err(String::from("Expected quantity")),
                }
            }
            Type::Varient(options) => {
                let mut idx = idx;
                let n_prim;
                (idx, n_prim) = Type::Quantity.value_from_bytes(ctx, idx, data).unwrap();

                match n_prim {
                    Primitive::Quantity(n) => {
                        let k = options.key(n);
                        let v;
                        (idx, v) = options
                            .get(k)
                            .unwrap()
                            .value_from_bytes(ctx, idx, data)
                            .unwrap();
                        Ok((idx, Primitive::Varient(k.clone(), Box::new(v))))
                    }
                    _ => Err(String::from("Expected quantity")),
                }
            }
        }
    }
}

impl Message for Type {
    fn get_type() -> Type {
        Type::Varient(Lookup::new(vec![
            (Key::from_str("named"), Type::Named(Key::from_str("key"))),
            (Key::from_str("byte"), Type::Tuple(vec![])),
            (Key::from_str("quantity"), Type::Tuple(vec![])),
            (
                Key::from_str("tuple"),
                Type::List(Box::new(Type::Named(Key::from_str("type")))),
            ),
            (Key::from_str("list"), Type::Named(Key::from_str("type"))),
            (
                Key::from_str("variant"),
                Type::Named(Key::from_str("lookup")),
            ),
        ]))
    }

    fn encode(&self) -> Primitive {
        match &self {
            Type::Named(key) => Primitive::Varient(Key::from_str("named"), Box::new(key.encode())),
            Type::Byte => {
                Primitive::Varient(Key::from_str("byte"), Box::new(Primitive::Vector(vec![])))
            }
            Type::Quantity => Primitive::Varient(
                Key::from_str("quantity"),
                Box::new(Primitive::Vector(vec![])),
            ),
            Type::Tuple(ts) => Primitive::Varient(
                Key::from_str("tuple"),
                Box::new(Primitive::Vector(ts.iter().map(|t| t.encode()).collect())),
            ),
            Type::List(t) => Primitive::Varient(Key::from_str("list"), Box::new(t.encode())),
            Type::Varient(options) => {
                Primitive::Varient(Key::from_str("variant"), Box::new(options.encode()))
            }
        }
    }

    fn decode(v: &Primitive) -> Result<Self, ()> {
        match v {
            Primitive::Varient(k, w) => {
                if k == &Key::from_str("named") {
                    if let Ok(key) = Key::decode(w) {
                        return Ok(Type::Named(key));
                    }
                } else if k == &Key::from_str("byte") {
                    if let Primitive::Vector(empty) = &**w {
                        if let 0 = empty.len() {
                            return Ok(Type::Byte);
                        }
                    }
                } else if k == &Key::from_str("quantity") {
                    if let Primitive::Vector(empty) = &**w {
                        if let 0 = empty.len() {
                            return Ok(Type::Quantity);
                        }
                    }
                } else if k == &Key::from_str("tuple") {
                    if let Primitive::Vector(ws) = &**w {
                        return Ok(Type::Tuple(
                            ws.iter().map(|w| Type::decode(w).unwrap()).collect(),
                        ));
                    }
                } else if k == &Key::from_str("list") {
                    return Ok(Type::List(Box::new(Type::decode(w).unwrap())));
                } else if k == &Key::from_str("variant") {
                    return Ok(Type::Varient(Lookup::decode(w).unwrap()));
                }
                Err(())
            }
            _ => Err(()),
        }
    }
}

fn get_meta_ctx() -> Lookup {
    return Lookup::new(vec![
        (Key::from_str("key"), Key::get_type()),
        (Key::from_str("lookup"), Lookup::get_type()),
        (Key::from_str("type"), Type::get_type()),
    ]);
}

fn main() {
    println!("awoo");
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_primitive_to_and_from_bytes() {
        for (v, t) in vec![
            (Primitive::Byte(123), Type::Byte),
            (Primitive::Quantity(10000000000), Type::Quantity),
            (
                Primitive::Vector(vec![
                    Primitive::Quantity(300),
                    Primitive::Byte(42),
                    Primitive::Quantity(600),
                ]),
                Type::Tuple(vec![Type::Quantity, Type::Byte, Type::Quantity]),
            ),
            (
                Primitive::Vector(vec![
                    Primitive::Quantity(300),
                    Primitive::Quantity(400),
                    Primitive::Quantity(500),
                    Primitive::Quantity(600),
                ]),
                Type::List(Box::new(Type::Quantity)),
            ),
            (
                Primitive::Vector((0..600).map(|i: usize| Primitive::Quantity(i)).collect()),
                Type::List(Box::new(Type::Quantity)),
            ),
            (
                Primitive::Varient(Key::from_str("thing2"), Box::new(Primitive::Quantity(400))),
                Type::Varient(Lookup::new(vec![
                    (Key::from_str("thing1"), Type::Byte),
                    (Key::from_str("thing2"), Type::Quantity),
                    (Key::from_str("thing3"), Type::Byte),
                ])),
            ),
            (
                Primitive::Varient(Key::from_str("sus596"), Box::new(Primitive::Quantity(500))),
                Type::Varient(Lookup::new(
                    (0..600)
                        .map(|i: usize| {
                            (
                                Key::from_string(String::from("sus") + &i.to_string()),
                                Type::Quantity,
                            )
                        })
                        .collect(),
                )),
            ),
        ] {
            let ctx = Lookup::new(vec![]);
            let d = t.value_to_bytes(&ctx, &v).unwrap();
            let (idx, w) = t.value_from_bytes(&ctx, 0, &d).unwrap();
            assert_eq!(idx, d.len());
            assert_eq!(v, w);
        }
    }

    #[test]
    fn test_named_to_and_from_bytes() {
        let ctx = Lookup::new(vec![
            (
                Key::from_str("foo"),
                Type::Varient(Lookup::new(vec![
                    (Key::from_str("first_foo"), Type::Byte),
                    (
                        Key::from_str("second_foo"),
                        Type::Named(Key::from_str("bar")),
                    ),
                ])),
            ),
            (
                Key::from_str("bar"),
                Type::Varient(Lookup::new(vec![
                    (Key::from_str("first_bar"), Type::Quantity),
                    (
                        Key::from_str("second_bar"),
                        Type::Named(Key::from_str("foo")),
                    ),
                ])),
            ),
        ]);
        let t = Type::Named(Key::from_str("foo"));

        let v = Primitive::Varient(
            Key::from_str("second_foo"),
            Box::new(Primitive::Varient(
                Key::from_str("second_bar"),
                Box::new(Primitive::Varient(
                    Key::from_str("first_foo"),
                    Box::new(Primitive::Byte(69)),
                )),
            )),
        );

        let data = t.value_to_bytes(&ctx, &v).unwrap();
        let (n, w) = t.value_from_bytes(&ctx, 0, &data).unwrap();
        assert_eq!(n, data.len());
        assert_eq!(v, w);
    }

    #[test]
    fn test_key_message_serialize_and_deserialize() {
        let ctx = Lookup::new(vec![]);
        let t = Key::get_type();

        //using encode -> value_to_bytes -> value_from_bytes -> decode
        let m = Key::from_str("this_is_a_test_key");
        println!("m = {:?}", m);
        let v = m.encode();
        println!("v = {:?}", v);
        let d = Key::get_type().value_to_bytes(&ctx, &v).unwrap();
        println!("d = {:?}", d);
        let (idx, w) = Key::get_type().value_from_bytes(&ctx, 0, &d).unwrap();
        assert_eq!(idx, d.len());
        assert_eq!(v, w);
        println!("w = {:?}", w);
        let n = Key::decode(&w).unwrap();
        println!("n = {:?}", n);
        assert_eq!(m, n);

        //using serialize -> deserialize
        let e = m.serialize(&ctx);
        println!("e = {:?}", e);
        assert_eq!(e, d);
        let o = Key::deserialize(&ctx, &e);
        println!("o = {:?}", o);
        assert_eq!(m, o);
    }

    #[test]
    fn test_lookup_message_serialize_and_deserialize() {
        let ctx = get_meta_ctx();
        let t = Lookup::get_type();

        //using encode -> value_to_bytes -> value_from_bytes -> decode
        let m = Lookup::new(vec![
            (Key::from_str("key_one"), Type::Byte),
            (Key::from_str("key_two"), Type::Quantity),
        ]);
        println!("m = {:?}", m);
        let v = m.encode();
        println!("v = {:?}", v);
        let d = Lookup::get_type().value_to_bytes(&ctx, &v).unwrap();
        println!("d = {:?}", d);
        let (idx, w) = Lookup::get_type().value_from_bytes(&ctx, 0, &d).unwrap();
        assert_eq!(idx, d.len());
        assert_eq!(v, w);
        println!("w = {:?}", w);
        let n = Lookup::decode(&w).unwrap();
        println!("n = {:?}", n);
        assert_eq!(m, n);

        //using serialize -> deserialize
        let e = m.serialize(&ctx);
        println!("e = {:?}", e);
        assert_eq!(e, d);
        let o = Lookup::deserialize(&ctx, &e);
        println!("o = {:?}", o);
        assert_eq!(m, o);
    }

    #[test]
    fn test_type_serialize_and_deserialize() {    
        let ctx = get_meta_ctx();
        println!("ctx = {:?}", ctx);
    
        for m in vec![
            Type::Named(Key::from_str("sus")),
            Type::Byte,
            Type::Quantity,
            Type::Tuple(vec![Type::Byte, Type::Quantity]),
            Type::List(Box::new(Type::Quantity)),
            Type::Varient(Lookup::new(vec![
                (Key::from_str("key_one"), Type::Byte),
                (Key::from_str("key_two"), Type::Quantity),
            ])),
        ] {
            println!("");
            println!("m = {:?}", m);
            let v = m.encode();
            println!("v = {:?}", v);
            let d = m.serialize(&ctx);
            println!("d = {:?}", d);
            let n = Type::deserialize(&ctx, &d);
            println!("n = {:?}", n);
            assert_eq!(m, n);
        }
    }

    #[test]
    fn test_meta_ctx_serialize_and_deserialize() {
        let ctx = get_meta_ctx();
        println!("ctx = {:?}", ctx);
        let d = ctx.serialize(&ctx);
        println!("d = {:?}", d);
        let ctx2 = Lookup::deserialize(&ctx, &d);
        println!("ctx2 = {:?}", ctx2);
        assert_eq!(ctx, ctx2);
    }
}
