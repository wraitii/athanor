use super::*;

impl Payload {
    fn capture_count(&self) -> usize {
        match self {
            Self::Closure(closure) => closure.captures.len(),
            _ => panic!("expected closure"),
        }
    }
}

#[test]
fn numeric_element_indices_respect_limits_and_exactness() {
    let cases = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -1.0,
        -0.0,
        0.0,
        0.5,
        1.0,
        1.5,
        4294967294.0,
        4294967295.0,
        4294967296.0,
        (usize::MAX as f64) - 4096.0,
        usize::MAX as f64,
    ];
    for limit in [u32::MAX as f64, usize::MAX as f64] {
        for number in cases {
            let expected =
                (number.is_finite() && number >= 0.0 && number < limit && number.fract() == 0.0)
                    .then_some(number as usize);
            assert_eq!(
                element_index(number, limit),
                expected,
                "{number} with limit {limit}"
            );
        }
    }
}

#[test]
fn enumeration_respects_hidden_shadowing_and_releases_receivers() {
    let heap = Heap::default();
    let prototype = heap.object().unwrap();
    let object = heap.object().unwrap();
    assert!(heap.set_property(prototype, "hidden", Value::TRUE));
    assert!(heap.set_property(prototype, "inherited", Value::TRUE));
    assert!(heap.define_property(object, "hidden", Value::TRUE, false));
    assert!(heap.set_property(object, "own", Value::TRUE));
    {
        let mut data = heap.0.borrow_mut();
        assert!(data.retain(prototype));
        data.set_prototype(heap_id(object).unwrap(), prototype);
    }
    let state = heap.enumerate(object).unwrap();
    heap.release(object);
    heap.release(prototype);
    let first = heap.enumerate_next(state).unwrap().unwrap();
    assert_eq!(heap.text(first).as_deref(), Some("own"));
    heap.release(first);
    let second = heap.enumerate_next(state).unwrap().unwrap();
    assert_eq!(heap.text(second).as_deref(), Some("inherited"));
    heap.release(second);
    assert_eq!(heap.enumerate_next(state), Some(None));
    heap.release(state);
    heap.drain(100);
    assert!(
        heap.0
            .borrow()
            .slots()
            .all(|slot| !matches!(slot.payload, Payload::Object(_)))
    );
    assert!(
        heap.0
            .borrow()
            .slots()
            .all(|slot| !matches!(slot.payload, Payload::Enumerator(_)))
    );
}

#[test]
fn unfinished_enumeration_releases_receiver_and_property_values() {
    let heap = Heap::default();
    let object = heap.object().unwrap();
    let child = heap.object().unwrap();
    assert!(heap.set_property(object, "child", child));
    let state = heap.enumerate(object).unwrap();
    heap.release(object);
    heap.release(child);
    heap.drain(100);
    assert!(heap.valid(object));
    assert!(heap.valid(child));
    heap.release(state);
    heap.drain(100);
    assert!(!heap.valid(object));
    assert!(!heap.valid(child));
    assert!(
        heap.0
            .borrow()
            .slots()
            .all(|slot| !matches!(slot.payload, Payload::Enumerator(_)))
    );
}

#[test]
fn regexp_cache_shares_code_without_retaining_objects() {
    let heap = Heap::default();
    let a = heap.regexp("x+", 0).unwrap();
    let b = heap.regexp("x+", 0).unwrap();
    assert_ne!(a, b);
    assert!(Rc::ptr_eq(
        &heap.regexp_compiled(a).unwrap(),
        &heap.regexp_compiled(b).unwrap()
    ));
    heap.release(a);
    heap.release(b);
    heap.drain(100);
    assert!(
        heap.0
            .borrow()
            .slots()
            .all(|slot| !matches!(slot.payload, Payload::RegExp(_)))
    );
    assert_eq!(heap.0.borrow().regexp_cache.len(), 1);
}

#[test]
fn shapes_track_layout_prototypes_and_fall_back_to_dictionaries() {
    let heap = Heap::default();
    let shape = |value: Value| heap.0.borrow().slot(heap_id(value).unwrap()).shape;
    let first = heap.object().unwrap();
    let second = heap.object().unwrap();
    assert_eq!(shape(first), shape(second));
    assert!(heap.set_property(first, "x", Value::TRUE));
    assert_ne!(shape(first), shape(second));
    assert!(heap.set_property(second, "x", Value::FALSE));
    assert_eq!(shape(first), shape(second));
    let layout = shape(first);
    assert!(heap.set_property(first, "x", Value::FALSE));
    assert!(heap.define_property(first, "x", Value::TRUE, true));
    assert_eq!(shape(first), layout);
    assert!(heap.define_property(first, "x", Value::TRUE, false));
    assert_ne!(shape(first), layout);
    assert_eq!(heap.own_properties(first), Some(Vec::new()));

    let prototype = heap.object().unwrap();
    {
        let mut data = heap.0.borrow_mut();
        assert!(data.retain(prototype));
        data.set_prototype(heap_id(second).unwrap(), prototype);
    }
    assert_ne!(shape(second), layout);
    assert_eq!(heap.property(second, "x"), Some(Value::FALSE));

    for index in 0..100 {
        assert!(heap.set_property(second, &format!("p{index}"), Value::number(index as f64)));
    }
    assert_eq!(shape(second), DICTIONARY);
    let names = heap.own_property_names(second, true).unwrap();
    assert_eq!(names.len(), 101);
    assert_eq!(names[..3], ["x", "p0", "p1"]);
    assert_eq!(heap.property(second, "p99"), Some(Value::number(99.0)));

    let indexed = heap.object().unwrap();
    assert!(heap.set_property(indexed, "b", Value::TRUE));
    assert!(heap.set_property(indexed, "1", Value::TRUE));
    assert_eq!(shape(indexed), DICTIONARY);
    assert_eq!(heap.own_property_names(indexed, true).unwrap(), ["1", "b"]);
    for value in [first, second, prototype, indexed] {
        heap.release(value);
    }
    heap.drain(1000);
    assert!(!heap.valid(prototype));
}

#[test]
fn array_presence_distinguishes_holes_and_undefined() {
    let heap = Heap::default();
    let array = heap.array(&[Value::UNDEFINED, Value::UNDEFINED]).unwrap();
    heap.mark_hole(array, 0);
    assert_eq!(heap.has_property(array, "0"), Some(false));
    assert_eq!(heap.has_property(array, "1"), Some(true));
    assert_eq!(heap.has_property(array, "2"), Some(false));
    heap.release(array);
    heap.drain(100);
}

#[test]
fn instances_retain_prototypes_and_release_their_properties() {
    let heap = Heap::default();
    let constructor = heap.closure(CodeId(1), &[]).unwrap();
    let prototype = heap.property(constructor, "prototype").unwrap();
    let child = heap.object().unwrap();
    assert!(heap.set_property(prototype, "child", child));
    heap.release(child);
    let instance = heap.constructed_object(constructor).unwrap();
    heap.release(constructor);
    heap.drain(100);
    assert!(heap.valid(prototype));
    assert_eq!(heap.property(instance, "child"), Some(child));
    heap.release(instance);
    heap.drain(100);
    assert!(!heap.valid(prototype));
    assert!(!heap.valid(child));
}

#[test]
fn arrow_and_generator_receivers_are_owned_and_released() {
    let heap = Heap::default();
    let receiver = heap.object().unwrap();
    let arrow = heap.closure(CodeId(1), &[]).unwrap();
    assert!(heap.bind_this(arrow, receiver));
    heap.release(receiver);
    assert_eq!(heap.lexical_this(arrow), Some(receiver));
    assert!(heap.valid(receiver));
    assert!(!heap.constructable(arrow));
    heap.release(arrow);
    heap.drain(100);
    assert!(!heap.valid(receiver));

    let receiver = heap.object().unwrap();
    let closure = heap.closure(CodeId(2), &[]).unwrap();
    let arguments = heap.array(&[]).unwrap();
    let generator = heap
        .generator(CodeId(3), closure, arguments, receiver)
        .unwrap();
    heap.release(receiver);
    heap.release(closure);
    heap.release(arguments);
    assert_eq!(heap.generator_receiver(generator), Some(receiver));
    heap.generator_complete(generator);
    assert!(!heap.valid(receiver));
    heap.release(generator);
    heap.drain(100);
    assert!(
        heap.0
            .borrow()
            .slots()
            .all(|slot| !matches!(slot.payload, Payload::Generator(_)))
    );
}

#[test]
fn abandoned_generators_release_suspended_frames_incrementally() {
    let heap = Heap::default();
    let closure = heap.closure(CodeId(1), &[]).unwrap();
    let argument = heap.string("argument").unwrap();
    let arguments = heap.array(&[argument]).unwrap();
    let generator = heap
        .generator(CodeId(2), closure, arguments, Value::UNDEFINED)
        .unwrap();
    heap.release(closure);
    heap.release(argument);
    heap.release(arguments);
    assert_eq!(heap.generator_begin(generator), Some((0, Vec::new())));
    assert_eq!(heap.generator_state(generator), Some(GENERATOR_RUNNING));
    assert!(heap.generator_begin(generator).is_none());
    let local = heap.array(&[]).unwrap();
    heap.generator_suspend(generator, 1, vec![local]);
    heap.release(generator);
    heap.drain(1);
    assert!(!heap.valid(local));
    assert!(heap.valid(closure));
    assert!(heap.valid(arguments));
    heap.drain(100);
    assert!(!heap.valid(closure));
    assert!(!heap.valid(arguments));
    assert!(!heap.valid(argument));
    assert!(
        heap.0
            .borrow()
            .slots()
            .all(|slot| !matches!(slot.payload, Payload::Generator(_)))
    );
}

#[test]
fn sets_retain_entries_and_clear_disposes_incrementally() {
    let heap = Heap::default();
    let set = heap.set_new().unwrap();
    let first = heap.array(&[]).unwrap();
    let second = heap.array(&[]).unwrap();
    assert!(heap.set_add(set, first));
    assert!(heap.set_add(set, first));
    assert!(heap.set_add(set, second));
    heap.release(first);
    heap.release(second);
    assert_eq!(heap.set_size(set), Some(2));
    assert!(heap.set_clear(set));
    assert_eq!(heap.set_size(set), Some(0));
    assert!(heap.valid(first));
    assert!(heap.valid(second));
    heap.drain(1);
    assert_ne!(heap.valid(first), heap.valid(second));
    heap.drain(100);
    assert!(!heap.valid(first));
    assert!(!heap.valid(second));
    let value = heap.string("é😀").unwrap();
    assert!(heap.set_add(set, value));
    heap.release(value);
    assert_eq!(heap.set_delete(set, value), Some(true));
    assert!(!heap.valid(value));
    heap.release(set);
    heap.drain(100);
    assert!(
        heap.0
            .borrow()
            .slots()
            .all(|slot| !matches!(slot.payload, Payload::Set(_)))
    );
}

#[test]
fn error_properties_own_values_and_dispose_incrementally() {
    let heap = Heap::default();
    let message = heap.string("oops").unwrap();
    let name = heap.string("Error").unwrap();
    let error = heap.error(message, name).unwrap();
    heap.release(message);
    heap.release(name);
    assert!(heap.set_property(error, "message", message));
    assert_eq!(heap.error_text(error).as_deref(), Some("Error: oops"));
    let child = heap.array(&[]).unwrap();
    assert!(heap.set_property(error, "extra", child));
    heap.release(child);
    heap.release(error);
    heap.drain(1);
    assert!(!heap.valid(error));
    assert!(matches!(
        heap.0.borrow().slot(heap_id(error).unwrap()).payload,
        Payload::Object(_)
    ));
    heap.drain(100);
    assert!(!heap.valid(message));
    assert!(!heap.valid(name));
    assert!(!heap.valid(child));
    assert!(
        heap.0
            .borrow()
            .slots()
            .all(|slot| !matches!(slot.payload, Payload::Object(_)))
    );
}

#[test]
fn closures_share_cells_and_retain_captured_values() {
    let heap = Heap::default();
    let first = heap.array(&[]).unwrap();
    let cell = heap.cell(first).unwrap();
    let left = heap.closure(CodeId(1), &[cell]).unwrap();
    let right = heap.closure(CodeId(2), &[cell]).unwrap();
    heap.release(first);
    heap.release(cell);
    let captured = heap.closure_parts(left).unwrap().1[0];
    let replacement = heap.array(&[]).unwrap();
    assert!(heap.cell_set(captured, replacement));
    heap.release(replacement);
    heap.drain(100);
    assert!(!heap.valid(first));
    assert_eq!(
        heap.cell_get(heap.closure_parts(right).unwrap().1[0]),
        Some(replacement)
    );
    assert!(heap.cell_set(captured, replacement));
    heap.release(left);
    heap.drain(100);
    assert!(heap.valid(replacement));
    heap.release(right);
    heap.drain(100);
    assert!(!heap.valid(captured));
    assert!(!heap.valid(replacement));
}

#[test]
fn closure_disposal_is_budgeted_by_capture_edges() {
    let heap = Heap::default();
    let cells: Vec<_> = (0..100)
        .map(|_| heap.cell(Value::UNDEFINED).unwrap())
        .collect();
    let closure = heap.closure(CodeId(1), &cells).unwrap();
    for cell in &cells {
        heap.release(*cell);
    }
    heap.release(closure);
    heap.drain(1);
    assert_eq!(
        heap.0
            .borrow()
            .slot(heap_id(closure).unwrap())
            .payload
            .capture_count(),
        99
    );
    assert!(cells.iter().any(|cell| heap.valid(*cell)));
    heap.drain(1000);
    assert!(cells.iter().all(|cell| !heap.valid(*cell)));
    assert!(
        heap.0
            .borrow()
            .slots()
            .all(|slot| !matches!(slot.payload, Payload::Closure(_)))
    );
    assert!(
        heap.0
            .borrow()
            .slots()
            .all(|slot| !matches!(slot.payload, Payload::Cell(_)))
    );
}

#[test]
fn closure_cell_cycle_does_not_keep_heap_alive() {
    let heap = Heap::default();
    let weak = Rc::downgrade(&heap.0);
    let cell = heap.cell(Value::UNDEFINED).unwrap();
    let closure = heap.closure(CodeId(1), &[cell]).unwrap();
    assert!(heap.cell_set(cell, closure));
    heap.release(cell);
    heap.release(closure);
    heap.drain(100);
    assert!(heap.valid(cell));
    assert!(heap.valid(closure));
    drop(heap);
    assert!(weak.upgrade().is_none());
}

#[test]
fn closure_rejects_invalid_captures() {
    let heap = Heap::default();
    let cell = heap.cell(Value::UNDEFINED).unwrap();
    let array = heap.array(&[]).unwrap();
    assert!(heap.closure(CodeId(1), &[array]).is_none());
    heap.release(cell);
    assert!(heap.closure(CodeId(1), &[cell]).is_none());
    heap.release(array);
}

#[test]
fn arrays_retain_edges_and_replace_safely() {
    let heap = Heap::default();
    let child = heap.array(&[]).unwrap();
    let parent = heap.array(&[child]).unwrap();
    heap.release(child);
    assert!(heap.set(parent, 0, child));
    heap.drain(100);
    assert_eq!(heap.get(parent, 0), Some(child));
    heap.release(parent);
    heap.drain(100);
    assert!(!heap.valid(parent));
    assert!(!heap.valid(child));
}

#[test]
fn destruction_is_iterative_and_budgeted() {
    let heap = Heap::default();
    let mut root = heap.array(&[]).unwrap();
    for _ in 0..10000 {
        let parent = heap.array(&[root]).unwrap();
        heap.release(root);
        root = parent;
    }
    heap.release(root);
    heap.drain(1);
    assert!(!heap.valid(root));
    assert!(!heap.retain(root));
    assert!(matches!(
        heap.0.borrow().slot(heap_id(root).unwrap()).payload,
        Payload::Array(_)
    ));
    heap.drain(30000);
    assert!(!heap.valid(root));
    assert!(
        heap.0
            .borrow()
            .slots()
            .all(|slot| !matches!(slot.payload, Payload::Array(_)))
    );
}

#[test]
fn host_root_keeps_array_alive_and_rejects_stale_tokens() {
    let heap = Heap::default();
    let value = heap.array(&[]).unwrap();
    let root = OwnedValue::from_owned(value, heap.clone());
    let copy = root.clone();
    drop(root);
    assert!(heap.valid(value));
    drop(copy);
    assert!(!heap.valid(value));
    assert!(!heap.retain(value));
    assert_eq!(heap.get(value, 0), None);
}

#[test]
fn pending_slots_are_reused_only_after_their_edges_are_released() {
    let heap = Heap::default();
    let child = heap.object().unwrap();
    let parent = heap.array(&vec![child; 100]).unwrap();
    heap.release(child);
    heap.release(parent);
    heap.drain(1);
    assert!(!heap.valid(parent));
    assert!(heap.valid(child));
    let other = heap.string("other").unwrap();
    assert_ne!(heap_id(parent), heap_id(other));
    assert!(heap.valid(child));
    heap.drain(1000);
    assert!(!heap.valid(child));
    heap.release(other);
    heap.drain(100);
}

#[test]
fn freed_slots_are_reused_across_kinds_and_allocation_churn_is_bounded() {
    let heap = Heap::default();
    let array = heap.array(&[]).unwrap();
    heap.release(array);
    heap.drain(100);
    let object = heap.object().unwrap();
    assert_eq!(heap_id(array), heap_id(object));
    assert!(heap.set_property(object, "value", Value::TRUE));
    heap.release(object);
    heap.drain(100);
    for _ in 0..1000 {
        let value = heap.string("temporary").unwrap();
        heap.release(value);
    }
    heap.drain(100);
    assert_eq!(heap.0.borrow().arena.len(), 1);
}
