class Message():
    @classmethod
    def get_type(cls):
        raise NotImplementedError(f"get_type not implemented for {cls}")

    def serialize(self, ctx):
        assert type(ctx) == Lookup
        return type(self).get_type().value_to_bytes(ctx, self.encode())

    @classmethod
    def deserialize(cls, ctx, data):
        assert type(ctx) == Lookup
        n, value = cls.get_type().value_from_bytes(ctx, 0, data)
        assert n == len(data)
        return cls.decode(value)
    
    def encode(self):
        raise NotImplementedError(f"encode not implemented for {type(self)}")

    @classmethod
    def decode(cls, value):
        raise NotImplementedError(f"decode not implemented for {cls}")


class Key(Message):
    ALLOWED_CHARS = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789"
    
    def __init__(self, name):
        assert type(name) == bytes
        assert len(name) > 0
        for x in name:
            assert x in list(type(self).ALLOWED_CHARS)
        self.name = name

    def __repr__(self):
        return f"Key({self.name})"

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)

    @classmethod
    def get_type(cls):
        return List(Byte())
    
    def encode(self):
        return [self.name[i:i+1] for i in range(len(self.name))]

    @classmethod
    def decode(cls, value):
        return cls(b"".join(value))


class Lookup(Message):
    def __init__(self, lookup):
        assert type(lookup) == dict
        for k, t in lookup.items():
            assert type(k) == Key
            assert isinstance(t, Type)
        self.lookup = lookup

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.items() == other.items()
        return False

    def __repr__(self):
        return "Lookup({" + ", ".join(repr(k) + ": " + repr(t) for k, t in self.items()) + "})"

    def __contains__(self, key):
        return key in self.lookup

    def __getitem__(self, key):
        return self.lookup[key]

    def __len__(self):
        return len(self.lookup)

    def index(self, key):
        return list(self.lookup.keys()).index(key)

    def key(self, n):
        return list(self.lookup.keys())[n]

    def items(self):
        return self.lookup.items()

    def keys(self):
        return self.lookup.keys()

    @classmethod
    def get_type(cls):
        return List(Tuple([Named(Key(b"key")), Named(Key(b"type"))]))
    
    def encode(self):
        return [[k.encode(), t.encode()] for k, t in self.items()]

    @classmethod
    def decode(cls, value):
        return cls({Key.decode(k) : Type.decode(t) for k, t in value})


class Type(Message):
    def __eq__(self, other):
        raise NotImplementedError(f"__eq__ not implemented for {type(self)}")
    
    def validate_value(self, ctx, v):
        assert type(ctx) == Lookup
        raise NotImplementedError(f"validate_value not implemented for {type(self)}")

    def value_to_bytes(self, ctx, v):
        assert type(ctx) == Lookup
        raise NotImplementedError(f"value_to_bytes not implemented for {type(self)}")

    def value_from_bytes(self, ctx, idx, data):
        assert type(ctx) == Lookup
        raise NotImplementedError(f"value_from_bytes not implemented for {type(self)}")

    @classmethod
    def get_type(cls):
        return Variant(Lookup({Key(b"named") : Named(Key(b"key")),
                               Key(b"byte") : Tuple([]),
                               Key(b"quantity") : Tuple([]),
                               Key(b"tuple") : List(Named(Key(b"type"))),
                               Key(b"list") : Named(Key(b"type")),
                               Key(b"variant") : Named(Key(b"lookup")),
                               Key(b"struct") : Named(Key(b"lookup"))}))

    def encode(self):
        return self.encode_impl()
    def encode_impl(self):
        raise NotImplementedError(f"encode_impl not implemented for {type(self)}")

    @classmethod
    def decode(cls, v):
        k, w = v
        return {Key(b"named") : Named,
                Key(b"byte") : Byte,
                Key(b"quantity") : Quantity,
                Key(b"tuple") : Tuple,
                Key(b"list") : List,
                Key(b"variant") : Variant,
                Key(b"struct") : Struct}[k].decode_impl(w)
    @classmethod
    def decode_impl(cls, v):
        raise NotImplementedError(f"decode_impl not implemented for {cls}")


class Named(Type):
    def __init__(self, key):
        assert type(key) == Key
        self.key = key

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.key == other.key
        return False

    def __repr__(self):
        return "Named(" + repr(self.key) + ")"

    def validate_value(self, ctx, v):
        assert type(ctx) == Lookup
        ctx[self.key].validate_value(ctx, v)

    def value_to_bytes(self, ctx, v):
        assert type(ctx) == Lookup
        return ctx[self.key].value_to_bytes(ctx, v)

    def value_from_bytes(self, ctx, idx, data):
        assert type(ctx) == Lookup
        return ctx[self.key].value_from_bytes(ctx, idx, data)

    def encode_impl(self):
        return Key(b"named"), self.key.encode()

    @classmethod
    def decode_impl(cls, v):
        return Named(Key.decode(v))

#encode Byte() as a length 1 byte string e.g. b"x"
class Byte(Type):
    def __init__(self):
        pass

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return True
        return False

    def __repr__(self):
        return "Byte()"

    def validate_value(self, ctx, v):
        assert type(ctx) == Lookup
        assert type(v) == bytes
        assert len(v) == 1

    def value_to_bytes(self, ctx, v):
        assert type(ctx) == Lookup
        return v

    def value_from_bytes(self, ctx, idx, data):
        assert type(ctx) == Lookup
        return idx + 1, data[idx:idx+1]

    def encode_impl(self):
        return Key(b"byte"), []

    @classmethod
    def decode_impl(cls, v):
        assert v == []
        return Byte()

#encode Quantity() as a non-negative int
class Quantity(Type):
    def __init__(self):
        pass

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return True
        return False

    def __repr__(self):
        return "Quantity()"

    def validate_value(self, ctx, v):
        assert type(ctx) == Lookup
        assert type(v) == int and v >= 0

    def value_to_bytes(self, ctx, v):
        assert type(ctx) == Lookup
        data = b""
        while v >= 128:
            data += bytes([(v % 128) + 128])
            v = v // 128
        data += bytes([v])
        return data

    def value_from_bytes(self, ctx, idx, data):
        assert type(ctx) == Lookup
        n = 0
        p = 0
        while True:
            idx, r = idx + 1, data[idx]
            n += (r % 128) << (7 * p)
            if r < 128:
                break
            p += 1
        return idx, n

    def encode_impl(self):
        return Key(b"quantity"), []

    @classmethod
    def decode_impl(cls, v):
        assert v == []
        return Quantity()

#encode Tuple([A, B, C]) as a list e.g. [a, b, c] where a, b, c are encoded values for A, B, C
class Tuple(Type):
    def __init__(self, ts):
        ts = tuple(ts)
        for t in ts:
            assert isinstance(t, Type)
        self.ts = ts

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.ts == other.ts
        return False

    def __repr__(self):
        return "Tuple([" + ", ".join(repr(t) for t in self.ts) + "])"

    def validate_value(self, ctx, v):
        assert type(ctx) == Lookup
        assert type(v) == list
        n = len(self.ts)
        assert len(v) == n
        for i in range(n):
            self.ts[i].validate_value(ctx, v[i])

    def value_to_bytes(self, ctx, v):
        assert type(ctx) == Lookup
        data = b""
        for t, w in zip(self.ts, v):
            data += t.value_to_bytes(ctx, w)
        return data

    def value_from_bytes(self, ctx, idx, data):
        assert type(ctx) == Lookup
        v = []
        for t in self.ts:
            idx, w = t.value_from_bytes(ctx, idx, data)
            v.append(w)
        return idx, v

    def encode_impl(self):
        return Key(b"tuple"), [t.encode() for t in self.ts]

    @classmethod
    def decode_impl(cls, v):
        return Tuple([Type.decode(t) for t in v])

#encode List(T) as a list e.g. [a, b, c, ...] where a, b, c, ... are encoded values for T
class List(Type):
    def __init__(self, t):
        assert isinstance(t, Type)
        self.t = t

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.t == other.t
        return False

    def __repr__(self):
        return "List(" + repr(self.t) + ")"

    def validate_value(self, ctx, v):
        assert type(ctx) == Lookup
        assert type(v) == list
        for w in v:
            self.t.validate_value(ctx, w)

    def value_to_bytes(self, ctx, v):
        assert type(ctx) == Lookup
        return Quantity().value_to_bytes(ctx, len(v)) + b"".join([self.t.value_to_bytes(ctx, w) for w in v])

    def value_from_bytes(self, ctx, idx, data):
        assert type(ctx) == Lookup
        idx, n = Quantity().value_from_bytes(ctx, idx, data)
        v = []
        for i in range(n):
            idx, w = self.t.value_from_bytes(ctx, idx, data)
            v.append(w)
        return idx, v

    def encode_impl(self):
        return Key(b"list"), self.t.encode()

    @classmethod
    def decode_impl(cls, v):
        return List(Type.decode(v))

#encode Variant({k : T, ...}) as a tuple (k, t) where t encodes a value of type T
class Variant(Type):
    def __init__(self, options):
        assert type(options) == Lookup
        self.options = options

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.options == other.options
        return False

    def __repr__(self):
        return "Variant({" + ", ".join(repr(k) + ": " + repr(t) for k, t in self.options.items()) + "})"

    def validate_value(self, ctx, v):
        assert type(ctx) == Lookup
        assert type(v) == tuple
        assert len(v) == 2
        k, w = v
        assert type(k) == Key
        assert k in self.options
        self.options[k].validate_value(ctx, w)

    def value_to_bytes(self, ctx, v):
        assert type(ctx) == Lookup
        k, w = v
        n = self.options.index(k)
        return Quantity().value_to_bytes(ctx, n) + self.options[k].value_to_bytes(ctx, w)

    def value_from_bytes(self, ctx, idx, data):
        assert type(ctx) == Lookup
        idx, n = Quantity().value_from_bytes(ctx, idx, data)
        k = self.options.key(n)
        idx, w = self.options[k].value_from_bytes(ctx, idx, data)
        return idx, (k, w)

    def encode_impl(self):
        return Key(b"variant"), self.options.encode()

    @classmethod
    def decode_impl(cls, v):
        return cls(Lookup.decode(v))

#encode Struct({k : T, ...}) as a dict {k : t, ...} where t encodes a value of type T
class Struct(Type):
    def __init__(self, attribs):
        assert type(attribs) == Lookup
        self.attribs = attribs

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.attribs == other.attribs
        return False

    def __repr__(self):
        return "Struct({" + ", ".join(repr(k) + ": " + repr(t) for k, t in self.attribs.items()) + "})"

    def validate_value(self, ctx, v):
        assert type(v) == dict
        assert set(v.keys()) == set(self.attribs.keys())
        for k, w in v.items():
            self.attribs[k].validate_value(ctx, w)

    def value_to_bytes(self, ctx, v):
        return Tuple([t for k, t in self.attribs.items()]).value_to_bytes(ctx, [v[k] for k in self.attribs.keys()])
 
    def value_from_bytes(self, ctx, idx, data):
        idx, v = Tuple([t for k, t in self.attribs.items()]).value_from_bytes(ctx, idx, data)
        return idx, {k : v[i] for i, k in enumerate(self.attribs.keys())}

    def encode_impl(self):
        return Key(b"struct"), self.attribs.encode()

    @classmethod
    def decode_impl(cls, v):
        return cls(Lookup.decode(v))



class Protocol():
    def __eq__(self, other):
        raise NotImplementedError(f"__eq__ not implemented for {type(self)}")
    
    @classmethod
    def get_type(cls):
        return Variant(Lookup({Key(b"function") : Tuple([Named(Key(b"type")), Named(Key(b"type"))]),
                               Key(b"call") : Tuple([Named(Key(b"type")), Named(Key(b"type"))])}))

    def encode(self):
        return self.encode_impl()
    def encode_impl(self):
        raise NotImplementedError(f"encode_impl not implemented for {type(self)}")

    @classmethod
    def decode(cls, v):
        k, w = v
        return {Key(b"function") : Function,
                Key(b"call") : Call}[k].decode_impl(w)
    @classmethod
    def decode_impl(cls, v):
        raise NotImplementedError(f"decode_impl not implemented for {cls}")


class Function(Protocol):
    def __init__(self, arg, ret):
        assert isinstance(arg, Type)
        assert isinstance(ret, Type)
        self.arg = arg
        self.ret = ret

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.arg == other.arg and self.ret == other.ret
        return False

    def __repr__(self):
        return f"Function({self.arg}, {self.ret})"

    def encode_impl(self):
        return Key(b"function"), [self.arg.encode(), self.ret.encode()]

    @classmethod
    def decode_impl(cls, v):
        assert type(v) == list
        assert len(v) == 2
        return cls(Type.decode(v[0]), Type.decode(v[1]))
    

class Call(Protocol):
    def __init__(self, arg, ret):
        assert isinstance(arg, Type)
        assert isinstance(ret, Type)
        self.arg = arg
        self.ret = ret

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.arg == other.arg and self.ret == other.ret
        return False

    def __repr__(self):
        return f"Call({self.arg}, {self.ret})"

    def encode_impl(self):
        return Key(b"call"), [self.arg.encode(), self.ret.encode()]

    @classmethod
    def decode_impl(cls, v):
        assert type(v) == list
        assert len(v) == 2
        return cls(Type.decode(v[0]), Type.decode(v[1]))




class Specification(Message):
    def __init__(self, ctx, protocols):
        assert type(ctx) == Lookup
        assert type(protocols) == dict
        for k, p in protocols.items():
            assert type(k) == Key
            assert isinstance(p, Protocol)
        self._ctx = ctx
        self._protocols = protocols

    @property
    def ctx(self):
        return self._ctx

    @property
    def protocols(self):
        return self._protocols

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.ctx == other.ctx and self.protocols == other.protocols
        return False

    def __repr__(self):
        return f"Specification({self.ctx}, {self.protocols})"

    @classmethod
    def get_type(cls):
        return Tuple([Lookup.get_type(), List(Tuple([Named(Key(b"key")), Named(Key(b"protocol"))]))])

    def encode(self):
        return [self.ctx.encode(), [[k.encode(), p.encode()] for k, p in self.protocols.items()]]

    @classmethod
    def decode(cls, v):
        assert type(v) == list
        assert len(v) == 2
        assert type(v[1]) == list
        for pair in v[1]:
            assert type(pair) == list
            assert len(pair) == 2
        return cls(Lookup.decode(v[0]), {Key.decode(k) : Protocol.decode(p) for k, p in v[1]})
    

META_CTX = Lookup({Key(b"key") : Key.get_type(),
                   Key(b"lookup") : Lookup.get_type(),
                   Key(b"type") : Type.get_type(),
                   Key(b"protocol") : Protocol.get_type(),
                   Key(b"spec") : Specification.get_type()})


def test():
    try:
        #basic types
        ctx = Lookup({})
        for v, t in [(b"x", Byte()),
                     (0, Quantity()),
                     (42, Quantity()),
                     (1000, Quantity()),
                     (10 ** 10, Quantity()),
                     ([42, b"y"], Tuple([Quantity(), Byte()])),
                     ([500, b"z", [500, b"w", 700]], Tuple([Quantity(), Byte(), Tuple([Quantity(), Byte(), Quantity()])])),
                     ([b"1", b"2", b"3", b"4"], List(Byte())),
                     ([100, 200, 300, 400, 500, 600, 700, 800], List(Quantity())),
                     ([], List(Quantity())),
                     ([3 * i for i in range(600)], List(Quantity())),
                     ((Key(b"thing2"), 600), Variant(Lookup({Key(b"thing1") : Byte(),
                                                             Key(b"thing2") : Quantity(),
                                                             Key(b"thing3") : Tuple([Byte(), Quantity()])}))),
                     ((Key(b"thing596"), [600, 700, 800, 900]), Variant(Lookup({Key(b"thing" + str(i).encode()) : Tuple([Quantity()] * (600-i)) for i in range(600)}))),
                     ({Key(b"b") : b"x", Key(b"a") : 400}, Struct(Lookup({Key(b"a") : Quantity(), Key(b"b") : Byte()})))]:
            print("v =", v)
            t.validate_value(ctx, v)
            d = t.value_to_bytes(ctx, v)
            print("d =", d)
            n, w = t.value_from_bytes(ctx, 0, d)
            print("w =", w)
            assert n == len(d)
            assert v == w
            print()

        #named types
        ctx = Lookup({Key(b"foo") : Variant(Lookup({Key(b"foo_one") : Byte(), Key(b"foo_two") : Named(Key(b"bar"))})),
                      Key(b"bar") : Variant(Lookup({Key(b"bar_one") : Quantity(), Key(b"bar_two") : Named(Key(b"foo"))}))})

        v = (Key(b"foo_two"), (Key(b"bar_two"), (Key(b"foo_one"), b"x")))
        print("v =", v)
        t = Named(Key(b"foo"))
        t.validate_value(ctx, v)
        d = t.value_to_bytes(ctx, v)
        print("d =", d)
        n, w = t.value_from_bytes(ctx, 0, d)
        print("w =", w)
        assert n == len(d)
        assert v == w
        print()

        #encoding and decoding
        for m in [Key(b"hello"),
                  Key(b"x" * 1000)]:
            assert isinstance(m, Message)
            print("m =", m)
            t = type(m).get_type()
            v = m.encode()
            print("v =", v)
            t.validate_value(Lookup({}), v)
            n = Key.decode(v)
            print("n =", n)
            assert m == n
            print()

        #meta ctx
        ctx = META_CTX
        
        #test_key_message_serialize_and_deserialize
        #key using encode -> value_to_bytes -> value_from_bytes -> decode
        m = Key(b"hello")
        print("m =", m)
        v = m.encode()
        print("v =", v)
        d = Named(Key(b"key")).value_to_bytes(ctx, v)
        print("d =", d)
        idx, w = Named(Key(b"key")).value_from_bytes(ctx, 0, d)
        print("w =", w)
        assert idx == len(d)
        assert v == w
        n = Key.decode(w)
        print("n =", n)
        assert m == n
        print()

        #key using serialize -> deserialize
        m = Key(b"hello")
        print("m =", m)
        d = m.serialize(ctx)
        print("d =", d)
        n = Key.decode(w)
        print("n =", n)
        assert m == n
        print()

        #lookup using serialize -> deserialize
        for m in [Lookup({}),
                  Lookup({Key(b"hello") : Byte()}),
                  Lookup({Key(b"foo") : Variant(Lookup({Key(b"foo_one") : Byte(), Key(b"foo_two") : Named(Key(b"bar"))})),
                          Key(b"bar") : Variant(Lookup({Key(b"bar_one") : Quantity(), Key(b"bar_two") : Named(Key(b"foo"))}))})]:
            print("m =", m)
            v = m.encode()
            print("v =", v)
            d = Named(Key(b"lookup")).value_to_bytes(ctx, v)
            print("d =", d)
            idx, w = Named(Key(b"lookup")).value_from_bytes(ctx, 0, d)
            print("w =", w)
            assert idx == len(d)
            assert v == w
            n = Lookup.decode(w)
            print("n =", n)
            assert m == n
            print()
        
            print("m =", m)
            d = m.serialize(ctx)
            print("d =", d)
            n = Lookup.decode(w)
            print("n =", n)
            assert m == n
            print()

        #test types serialize -> deserialize
        ctx = META_CTX
        for m in [Named(Key(b"sus")),
                  Byte(),
                  Quantity(),
                  Tuple([Byte(), Quantity()]),
                  List(Quantity()),
                  Variant(Lookup({Key(b"key_one") : Byte(),
                                  Key(b"key_two") : Quantity()})),
                  Struct(Lookup({Key(b"key_one") : Byte(),
                                 Key(b"key_two") : Quantity()}))]:
            print("m =", m)
            d = m.serialize(ctx)
            print("d =", d)
            n = Type.deserialize(ctx, d)
            print("n =", n)
            assert m == n
            print()

        #meta ctx serialize -> deserialize
        ctx = META_CTX
        m = META_CTX
        print("m =", m)
        d = m.serialize(ctx)
        print("d =", d)
        n = Lookup.deserialize(ctx, d)
        print("n =", n)
        assert m == n
        
    except AssertionError as e:
        print("TESTS FAILED...")
        raise e
    else:
        print("TESTS PASSED :)")



import socket
import time
import threading

class Connection():
    def __init__(self, sock, spec):
        assert isinstance(sock, socket.socket)
        assert type(spec) == Specification
        
        self._sock = sock
        self._sock.setblocking(False)
        self._buf = b""

        self._spec = spec

        #handshake
        #1) check that metacontexts agree
        self._send_dgram(META_CTX.serialize(META_CTX))
        other_META_CTX = Lookup.deserialize(META_CTX, self._recv_dgram())
        if META_CTX != other_META_CTX:
            raise Exception("Meta contexts do not agree")
        
        #2) use the meta specification handshake function to exchange specifications 
        self._send_dgram(self._spec.serialize(META_CTX))
        other_spec = Specification.deserialize(META_CTX, self._recv_dgram())

        #3) check that contexts agree
        if self._spec.ctx != other_spec.ctx:
            raise Exception("Contexts do not agree")

        #4) check that message flows are compatible
        
        import random
        time.sleep(random.uniform(0, 0.5))
        
        print(threading.get_ident(), self._spec)
        print(threading.get_ident(), other_spec)

    def _send_dgram(self, data):
        self._sock.sendall(Quantity().value_to_bytes(self._spec.ctx, len(data)) + data)

    def _recv_dgrams(self, num_dgram):
        assert type(num_dgram) == int
        assert num_dgram >= 0

        dgrams = []
        
        #recv as much as possible and put it in self._buf
        while True:
            try:
                self._buf += self._sock.recv(4096)
            except BlockingIOError:
                #nothing left to recv
                break
            
        #read dgrams from self._buf
        while num_dgram > 0:
            #check that self._buf has enough bytes to store a quantity
            for b in self._buf:
                if b // 128 == 0:
                    break
            else:
                return None
            
            #parse the quantity
            idx, n = Quantity().value_from_bytes(self._spec.ctx, 0, self._buf)
            #check if we have the whole dgram
            if idx + n <= len(self._buf):
                dgram = self._buf[idx:idx+n]
                self._buf = self._buf[idx+n:] #remove the dgram from self._buf
                dgrams.append(dgram)
            else:
                return None
            num_dgram -= 1

        return dgrams

    def _recv_dgram(self):
        while True:
            dgrams = self._recv_dgrams(1)
            if not dgrams is None:
                return dgrams[0]
            time.sleep(0.1)
                


        

def test_socket():    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 5000))
    s.listen()

    t = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    t.connect(("127.0.0.1", 5000))

    s, _ = s.accept()

    def run_a():
        a = Connection(s, Specification(Lookup({}),
                                        {Key(b"flub") : Function(Byte(), Byte()),
                                         Key(b"floobe") : Function(Quantity(), Quantity())}))

    def run_b():
        b = Connection(t, Specification(Lookup({}),
                                        {Key(b"flub") : Call(Byte(), Byte())}))
        

    thread_a = threading.Thread(target = run_a)
    thread_b = threading.Thread(target = run_b)

    thread_a.start()
    thread_b.start()

    thread_a.join()
    thread_b.join()

    print("done")




if __name__ == "__main__":
    test()
##    d = META_SPEC.serialize(META_SPEC.ctx)
##    print([b for b in d])
##    test_socket()





















    



