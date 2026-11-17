use super::*;

#[test]
fn native_properties_retain_values_and_participate_in_cycle_analysis() {
    let heap = Heap::default();
    let native = heap.native_function(NativeFunction::Object).unwrap();
    let child = heap.object().unwrap();
    assert!(heap.set_property(native, "child", child));
    heap.release(child);
    heap.release(native);

    let stats = heap.cycle_stats();
    assert_eq!((stats.live, stats.unreachable), (2, 0));
    assert_eq!(heap.property(native, "child"), Some(child));
    assert!(heap.set_property(native, "child", Value::UNDEFINED));
    heap.drain(100);
    assert!(!heap.valid(child));
    let stats = heap.cycle_stats();
    assert_eq!((stats.live, stats.unreachable), (1, 0));
}

#[test]
fn array_buffers_wait_for_disposal_and_reset_holes() {
    let heap = Heap::default();
    let child = heap.string("owned element").unwrap();
    let array = heap.array(&[child; 10]).unwrap();
    let pointer = heap.0.borrow().slot(heap_id(array).unwrap()).data;
    heap.release(child);
    heap.release(array);
    heap.drain(1);
    assert!(heap.valid(child));
    let buffer = heap.0.borrow_mut().array_buffers.take(10).unwrap();
    assert_ne!(buffer.as_ptr().cast::<u8>(), pointer);
    heap.drain(100);
    assert!(!heap.valid(child));
    let reused = heap.array_of_length(10).unwrap();
    assert_eq!(heap.0.borrow().slot(heap_id(reused).unwrap()).data, pointer);
    assert_eq!(heap.len(reused), Some(10));
    for index in 0..10 {
        assert!(!heap.present(reused, index));
        assert_eq!(heap.get(reused, index), Some(Value::UNDEFINED));
    }
    heap.release(reused);
    heap.drain(100);
    let dense = heap.array(&[Value::TRUE; 10]).unwrap();
    assert_eq!(heap.0.borrow().slot(heap_id(dense).unwrap()).data, pointer);
    assert_eq!(
        heap.0.borrow().slot(heap_id(dense).unwrap()).kind,
        kind::ARRAY
    );
    assert!(heap.present(dense, 9));
    assert_eq!(heap.get(dense, 9), Some(Value::TRUE));
    heap.release(dense);
}

#[test]
fn array_buffers_with_irregular_capacities_fit_reused_requests() {
    let heap = Heap::default();
    let elements = vec![Value::TRUE; 10];
    let pointer = elements.as_ptr().cast::<u8>();
    let array = heap.array_from_owned(elements).unwrap();
    heap.release(array);
    heap.drain(100);
    let larger = heap.array(&[Value::FALSE; 11]).unwrap();
    assert_ne!(heap.0.borrow().slot(heap_id(larger).unwrap()).data, pointer);
    let smaller = heap.array(&[Value::TRUE; 8]).unwrap();
    assert_eq!(
        heap.0.borrow().slot(heap_id(smaller).unwrap()).data,
        pointer
    );
    assert_eq!(heap.len(larger), Some(11));
    assert_eq!(heap.get(larger, 10), Some(Value::FALSE));
    assert_eq!(heap.len(smaller), Some(8));
    heap.release(larger);
    heap.release(smaller);
}

#[test]
fn string_buffers_reset_contents_and_preserve_unicode() {
    let heap = Heap::default();
    let source = heap.string("étoile🌟").unwrap();
    let pointer = heap
        .0
        .borrow()
        .string(heap_id(source).unwrap())
        .unwrap()
        .as_ptr();
    heap.release(source);
    heap.drain(1);
    let reused = heap.string("雪と海の月").unwrap();
    assert_eq!(heap.text(reused).as_deref(), Some("雪と海の月"));
    assert_eq!(
        heap.0
            .borrow()
            .string(heap_id(reused).unwrap())
            .unwrap()
            .as_ptr(),
        pointer
    );
    let suffix = heap.string("🌟").unwrap();
    let joined = heap.concat_strings(reused, suffix).unwrap();
    assert_eq!(heap.text(joined).as_deref(), Some("雪と海の月🌟"));
    heap.release(joined);
    heap.drain(100);
    let parts = heap.concat_string_parts(&[suffix, reused, suffix]).unwrap();
    assert_eq!(heap.text(parts).as_deref(), Some("🌟雪と海の月🌟"));
    heap.release(parts);
    heap.release(reused);
    heap.release(suffix);
    heap.drain(100);
}

#[test]
fn cycle_stats_distinguish_rooted_and_unreachable_self_cycles() {
    let heap = Heap::default();
    let object = heap.object().unwrap();
    assert!(heap.set_property(object, "self", object));

    let stats = heap.cycle_stats();
    assert_eq!((stats.live, stats.unreachable), (1, 0));
    assert_eq!((stats.cyclic_components, stats.cyclic_nodes), (1, 1));
    assert_eq!(
        (stats.unreachable_components, stats.unreachable_cyclic_nodes),
        (0, 0)
    );

    heap.release(object);
    let stats = heap.cycle_stats();
    assert_eq!((stats.live, stats.unreachable), (1, 1));
    assert_eq!((stats.cyclic_components, stats.cyclic_nodes), (1, 1));
    assert_eq!(
        (stats.unreachable_components, stats.unreachable_cyclic_nodes),
        (1, 1)
    );
}

#[test]
fn cycle_stats_count_duplicate_edges_and_acyclic_tails() {
    let heap = Heap::default();
    let first = heap.object().unwrap();
    let second = heap.object().unwrap();
    let tail = heap.object().unwrap();
    assert!(heap.set_property(first, "second", second));
    assert!(heap.set_property(first, "duplicate", second));
    assert!(heap.set_property(second, "first", first));
    assert!(heap.set_property(second, "tail", tail));
    heap.release(first);
    heap.release(second);
    heap.release(tail);

    let stats = heap.cycle_stats();
    assert_eq!((stats.live, stats.unreachable), (3, 3));
    assert_eq!((stats.cyclic_components, stats.cyclic_nodes), (1, 2));
    assert_eq!(
        (stats.unreachable_components, stats.unreachable_cyclic_nodes),
        (1, 2)
    );
}

#[test]
fn cycle_stats_include_closure_cells() {
    let heap = Heap::default();
    let cell = heap.cell(Value::UNDEFINED).unwrap();
    let closure = heap.closure(CodeId(1), &[cell]).unwrap();
    assert!(heap.cell_set(cell, closure));
    heap.release(cell);
    heap.release(closure);

    let stats = heap.cycle_stats();
    assert_eq!((stats.live, stats.unreachable), (2, 2));
    assert_eq!((stats.cyclic_components, stats.cyclic_nodes), (1, 2));
    assert_eq!(
        (stats.unreachable_components, stats.unreachable_cyclic_nodes),
        (1, 2)
    );
}

#[test]
fn cycle_stats_drain_pending_disposals() {
    let heap = Heap::default();
    let cycle = heap.object().unwrap();
    assert!(heap.set_property(cycle, "self", cycle));
    let holder = heap.array(&[cycle]).unwrap();
    heap.release(cycle);
    heap.release(holder);

    let stats = heap.cycle_stats();
    assert_eq!((stats.live, stats.unreachable), (1, 1));
    assert_eq!((stats.cyclic_components, stats.cyclic_nodes), (1, 1));
    assert_eq!(
        (stats.unreachable_components, stats.unreachable_cyclic_nodes),
        (1, 1)
    );
}

#[test]
fn split_results_own_reused_source() {
    let heap = Heap::default();
    let source = heap.string("terrain/grass").unwrap();
    let separator = heap.string("|").unwrap();
    let mut values = Vec::new();
    heap.split_string_into(source, Some(separator), &mut values)
        .unwrap();
    assert_eq!(values, vec![source]);
    let array = heap.array_from_owned(values).unwrap();
    heap.release(source);
    heap.release(separator);
    heap.drain(100);
    assert_eq!(heap.text(source).as_deref(), Some("terrain/grass"));
    heap.release(array);
    heap.drain(100);
    assert!(!heap.valid(source));
}

impl Payload {
    fn capture_count(&self) -> usize {
        match self {
            Self::Closure(closure) => closure.captures.len(),
            _ => panic!("expected closure"),
        }
    }
}

#[test]
fn small_objects_reuse_pending_heads_and_release_owned_edges() {
    let heap = Heap::default();
    let prototype = heap.object().unwrap();
    let child = heap.array(&[]).unwrap();
    let message = heap.string("message").unwrap();
    let name = heap.string("Error").unwrap();
    let object = heap.error(message, name).unwrap();
    assert!(heap.set_property(object, "child", child));
    {
        let mut data = heap.0.borrow_mut();
        assert!(data.retain(prototype));
        data.set_prototype(heap_id(object).unwrap(), prototype);
    }
    for value in [prototype, child, message, name, object] {
        heap.release(value);
    }
    let pointer = heap.0.borrow().slot(heap_id(object).unwrap()).data;
    let reused = heap.object().unwrap();
    assert_eq!(reused, object);
    assert_eq!(heap.property(reused, "child"), Some(Value::UNDEFINED));
    assert_eq!(
        heap.own_property_names(reused, false).unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(heap.error_text(reused), None);
    {
        let data = heap.0.borrow();
        let slot = data.slot(heap_id(reused).unwrap());
        assert_eq!(slot.data, pointer);
        assert_eq!(slot.length, 0);
        assert_eq!(slot.references, 1);
        assert_eq!(
            data.object(heap_id(reused).unwrap()).unwrap().prototype,
            None
        );
        for value in [prototype, child, message, name] {
            let id = heap_id(value).unwrap();
            assert_eq!(data.slot(id).references, 0);
            assert!(data.pending.contains(&id));
        }
        assert!(!data.pending.contains(&heap_id(reused).unwrap()));
    }
    assert!(heap.set_property(reused, "fresh", Value::TRUE));
    heap.drain(100);
    assert_eq!(heap.property(reused, "fresh"), Some(Value::TRUE));
    heap.release(reused);
    heap.drain(100);
    assert!(!heap.valid(reused));
}

#[test]
fn small_object_reuse_handles_partial_disposal() {
    let heap = Heap::default();
    let child = heap.string("child").unwrap();
    let object = heap.object().unwrap();
    assert!(heap.set_property(object, "child", child));
    assert!(heap.set_property(object, "last", Value::TRUE));
    heap.release(child);
    heap.release(object);
    heap.drain(1);
    let reused = heap.object().unwrap();
    assert_eq!(reused, object);
    assert!(!heap.valid(child));
    assert_eq!(heap.property(reused, "child"), Some(Value::UNDEFINED));
    heap.drain(100);
    assert!(heap.valid(reused));
    heap.release(reused);
    heap.drain(100);
}

#[test]
fn small_objects_upgrade_without_changing_identity_or_ownership() {
    let heap = Heap::default();
    let prototype = heap.object().unwrap();
    let child = heap.string("child").unwrap();
    assert!(heap.set_property(prototype, "inherited", child));
    let object = heap.object_with_prototype(prototype).unwrap();
    assert!(heap.define_property(object, "child", child, false));
    heap.release(child);
    heap.release(prototype);
    for index in 1..INLINE_PROPERTIES {
        assert!(heap.set_property(object, &format!("p{index}"), Value::number(index as f64)));
    }
    let id = heap_id(object).unwrap();
    let inline_data = {
        let data = heap.0.borrow();
        let slot = data.slot(id);
        assert_eq!(slot.kind, kind::OBJECT | kind::INLINE);
        assert_eq!(slot.length, INLINE_PROPERTIES);
        let address = slot.data as usize;
        assert!((id as usize..id as usize + arena::SLOT_SIZE).contains(&address));
        assert_eq!(unsafe { *slot.data.cast::<Value>() }, child);
        slot.data
    };
    assert!(heap.set_property(object, "overflow", Value::TRUE));
    {
        let data = heap.0.borrow();
        let slot = data.slot(id);
        assert_eq!(slot.kind, kind::OBJECT);
        assert_eq!(slot.length, INLINE_PROPERTIES + 1);
        assert_ne!(slot.data, inline_data);
        assert_eq!(unsafe { *slot.data.cast::<Value>() }, child);
        assert_eq!(data.slot(heap_id(child).unwrap()).references, 2);
        assert_eq!(data.slot(heap_id(prototype).unwrap()).references, 1);
    }
    assert_eq!(heap.property(object, "inherited"), Some(child));
    assert_eq!(heap.property(object, "child"), Some(child));
    assert_eq!(heap.has_property(object, "overflow"), Some(true));
    assert!(
        !heap
            .own_property_names(object, true)
            .unwrap()
            .contains(&"child".into())
    );
    heap.release(object);
    heap.drain(100);
    assert!(!heap.valid(object));
    assert!(!heap.valid(prototype));
    assert!(!heap.valid(child));
}

#[test]
fn small_objects_convert_to_dictionaries_and_release_values() {
    let heap = Heap::default();
    let object = heap.object().unwrap();
    let child = heap.string("child").unwrap();
    assert!(heap.set_property(object, "named", child));
    assert!(heap.set_property(object, "2", child));
    heap.release(child);
    {
        let data = heap.0.borrow();
        let slot = data.slot(heap_id(object).unwrap());
        assert_eq!(slot.kind, kind::OBJECT);
        assert_eq!(slot.shape, DICTIONARY);
        assert!(slot.data.is_null());
    }
    assert_eq!(heap.property(object, "named"), Some(child));
    assert_eq!(heap.property(object, "2"), Some(child));
    assert_eq!(
        heap.own_property_names(object, true).unwrap(),
        ["2", "named"]
    );
    heap.release(object);
    heap.drain(100);
    assert!(!heap.valid(child));
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
            .all(|slot| !matches!(slot.payload, Payload::Object(_) | Payload::SmallObject(_)))
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
        .generator(generator_plan(3), closure, arguments, receiver)
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
        .generator(generator_plan(2), closure, arguments, Value::UNDEFINED)
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
        Payload::Object(_) | Payload::SmallObject(_)
    ));
    heap.drain(100);
    assert!(!heap.valid(message));
    assert!(!heap.valid(name));
    assert!(!heap.valid(child));
    assert!(
        heap.0
            .borrow()
            .slots()
            .all(|slot| !matches!(slot.payload, Payload::Object(_) | Payload::SmallObject(_)))
    );
}

#[test]
fn cell_data_tracks_values_at_a_stable_address() {
    let heap = Heap::default();
    let cell = heap.cell(Value::UNINITIALIZED).unwrap();
    let address = heap
        .0
        .borrow()
        .slot(heap_id(cell).unwrap())
        .data
        .cast::<Value>();
    assert_eq!(unsafe { *address }, Value::UNINITIALIZED);
    assert!(heap.cell_set(cell, Value::number(17.0)));
    for _ in 0..1000 {
        let object = heap.object().unwrap();
        heap.release(object);
    }
    assert_eq!(unsafe { *address }, Value::number(17.0));
    assert_eq!(
        heap.0
            .borrow()
            .slot(heap_id(cell).unwrap())
            .data
            .cast::<Value>(),
        address
    );
    heap.release(cell);
    heap.drain(1000);
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

#[test]
fn bulk_array_extension_retains_values_and_refreshes_storage() {
    let heap = Heap::default();
    let child = heap.string("texture").unwrap();
    let source = heap.array(&[child, HOLE]).unwrap();
    heap.release(child);
    let destination = heap.array(&[]).unwrap();
    let values = heap.owned_indexed_values(source).unwrap();
    assert!(heap.extend_array_values(destination, &values));
    heap.release_all(&values);
    let values = heap.owned_indexed_values(destination).unwrap();
    assert!(heap.extend_array_values(destination, &values));
    heap.release_all(&values);
    heap.release(source);
    heap.drain(1000);
    assert_eq!(heap.len(destination), Some(4));
    assert_eq!(heap.get(destination, 0), Some(child));
    assert_eq!(heap.get(destination, 1), Some(Value::UNDEFINED));
    assert_eq!(heap.get(destination, 2), Some(child));
    assert_eq!(heap.get(destination, 3), Some(Value::UNDEFINED));
    let data = heap.0.borrow();
    let slot = data.slot(heap_id(destination).unwrap());
    assert_eq!(slot.length, 4);
    assert_eq!(slot.kind, kind::ARRAY);
    assert_eq!(
        slot.data,
        data.array(heap_id(destination).unwrap())
            .unwrap()
            .elements
            .as_ptr()
            .cast()
    );
    drop(data);
    heap.release(destination);
    heap.drain(1000);
    assert!(!heap.valid(child));
}

#[test]
fn owned_array_and_snapshot_balance_references() {
    let heap = Heap::default();
    let child = heap.string("texture").unwrap();
    let array = heap.array_from_owned(vec![child]).unwrap();
    let values = heap.owned_indexed_values(array).unwrap();
    heap.release(array);
    heap.drain(1000);
    assert_eq!(heap.text(values[0]).as_deref(), Some("texture"));
    heap.release_all(&values);
    heap.drain(1000);
    assert!(!heap.valid(child));
}

#[test]
fn disposal_respects_budget_across_shared_array_edges() {
    let heap = Heap::default();
    let child = heap.string("shared").unwrap();
    let first = heap.array(&vec![child; 100]).unwrap();
    let second = heap.array(&vec![child; 100]).unwrap();
    heap.release(child);
    heap.release(first);
    heap.release(second);
    heap.drain(17);
    let data = heap.0.borrow();
    let remaining: usize = [first, second]
        .into_iter()
        .map(|value| match &data.slot(heap_id(value).unwrap()).payload {
            Payload::Array(array) => array.elements.len(),
            _ => panic!("array was disposed before its edges"),
        })
        .sum();
    assert_eq!(remaining, 183);
    assert_eq!(data.slot(heap_id(child).unwrap()).references, 183);
    drop(data);
    heap.drain(1000);
    assert!(!heap.valid(child));
}

#[test]
fn instances_reuse_slots_with_the_requested_prototype() {
    let heap = Heap::default();
    let first = heap.object().unwrap();
    let second = heap.object().unwrap();
    assert!(heap.set_property(first, "inherited", Value::TRUE));
    assert!(heap.set_property(second, "inherited", Value::FALSE));
    let mut previous = None;
    for prototype in [first, first, second, Value::UNDEFINED] {
        let object = heap.object_with_prototype(prototype).unwrap();
        if let Some(previous) = previous {
            assert_eq!(object, previous);
        }
        let expected = if prototype == Value::UNDEFINED {
            Value::UNDEFINED
        } else {
            heap.property(prototype, "inherited").unwrap()
        };
        assert_eq!(heap.property(object, "inherited"), Some(expected));
        {
            let mut data = heap.0.borrow_mut();
            let prototype = prototype.is_object().then_some(prototype);
            let root = data.shapes.root(prototype);
            let slot = data.slot(heap_id(object).unwrap());
            assert_eq!(slot.shape, root);
            assert_eq!(slot.length, 0);
            assert_eq!(slot.references, 1);
            for value in [first, second] {
                let references = 1 + u32::from(prototype == Some(value));
                assert_eq!(data.slot(heap_id(value).unwrap()).references, references);
            }
        }
        assert!(heap.set_property(object, "old", Value::TRUE));
        heap.release(object);
        previous = Some(object);
    }
    heap.release(first);
    heap.release(second);
    heap.drain(1000);
    assert!(!heap.valid(first));
    assert!(!heap.valid(second));
}

#[test]
fn cached_additions_preserve_layout_enumerability_and_owned_values() {
    let heap = Heap::default();
    let prototype = heap.object().unwrap();
    let mut cache = PropertyWriteCache::default();
    for padding in [0, 7, 8, 63, 64] {
        for prototype in [Value::UNDEFINED, prototype] {
            for enumerable in [true, true, false, false, true] {
                let object = heap.object_with_prototype(prototype).unwrap();
                for index in 0..padding {
                    assert!(heap.set_property(object, &format!("p{index}"), Value::TRUE));
                }
                let child = heap.array(&[]).unwrap();
                assert!(heap.write_property(
                    object,
                    "child",
                    child,
                    Some(enumerable),
                    Some(&mut cache)
                ));
                heap.release(child);
                assert_eq!(heap.property(object, "child"), Some(child));
                assert_eq!(
                    heap.own_property_names(object, true)
                        .unwrap()
                        .contains(&"child".to_owned()),
                    enumerable
                );
                assert!(heap.set_property_cached(object, "child", Value::FALSE, &mut cache));
                heap.drain(1000);
                assert!(!heap.valid(child));
                assert_eq!(heap.property(object, "child"), Some(Value::FALSE));
                heap.release(object);
            }
        }
    }
    heap.release(prototype);
    heap.drain(1000);
    assert!(!heap.valid(prototype));
}

#[test]
fn property_write_cache_follows_heap_identity() {
    let mut cache = PropertyWriteCache::default();
    for padding in [0, 5, 1] {
        let heap = Heap::default();
        let object = heap.object().unwrap();
        for index in 0..padding {
            heap.set_property(object, &format!("p{index}"), Value::TRUE);
        }
        let child = heap.object().unwrap();
        assert!(heap.set_property_cached(object, "child", child, &mut cache));
        heap.release(child);
        assert_eq!(heap.property(object, "child"), Some(child));
        assert!(heap.set_property_cached(object, "child", Value::FALSE, &mut cache));
        heap.drain(1000);
        assert!(!heap.valid(child));
        assert_eq!(heap.property(object, "child"), Some(Value::FALSE));
        heap.release(object);
        heap.drain(1000);
    }
}

#[test]
fn jit_cached_additions_retain_and_destroy_children() {
    for ssa_opt in [true, false] {
        let mut engine = crate::Engine::with_options(crate::EngineOptions {
            ssa_opt,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                "function make(child) { const o = {}; o.child = child; return o; }",
                "write_cache.js",
            )
            .unwrap();
        let heap = engine.global().heap().clone();
        for _ in 0..100 {
            let child = heap.array(&[]).unwrap();
            let result = engine.call("make", &[child]).unwrap();
            let object = result.value();
            assert_eq!(heap.property(object, "child"), Some(child));
            assert_eq!(heap.0.borrow().slot(heap_id(child).unwrap()).references, 2);
            assert_eq!(heap.0.borrow().slot(heap_id(object).unwrap()).length, 1);
            heap.release(child);
            drop(result);
            heap.drain(1000);
            assert!(!heap.valid(object));
            assert!(!heap.valid(child));
        }
    }
}

#[test]
fn jit_cached_additions_guard_representation_and_errors() {
    for ssa_opt in [true, false] {
        let mut engine = crate::Engine::with_options(crate::EngineOptions {
            ssa_opt,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                "function write(o, child) { o.child = child; }",
                "write_cache.js",
            )
            .unwrap();
        let heap = engine.global().heap().clone();
        let warm = heap.object().unwrap();
        engine.call("write", &[warm, Value::TRUE]).unwrap();
        let boxed = heap.object().unwrap();
        {
            let mut data = heap.0.borrow_mut();
            let id = heap_id(boxed).unwrap();
            let slot = data.slot_mut(id);
            let Payload::SmallObject(object) = &slot.payload else {
                panic!()
            };
            slot.payload = Payload::Object(Object {
                properties: Properties::Shaped(Vec::new()),
                error: object.error,
                prototype: object.prototype(),
            });
            data.sync(id);
        }
        let child = heap.array(&[]).unwrap();
        engine.call("write", &[boxed, child]).unwrap();
        assert_eq!(heap.property(boxed, "child"), Some(child));
        assert_eq!(heap.0.borrow().slot(heap_id(child).unwrap()).references, 2);
        for receiver in [Value::NULL, Value::number(1.0)] {
            assert!(engine.call("write", &[receiver, child]).is_err());
            assert_eq!(heap.0.borrow().slot(heap_id(child).unwrap()).references, 2);
        }
        engine.call("write", &[child, warm]).unwrap();
        assert_eq!(heap.property(child, "child"), Some(warm));
        heap.release(child);
        heap.release(warm);
        heap.release(boxed);
        heap.drain(1000);
        assert!(!heap.valid(child));
    }
}

fn generator_plan(code: u64) -> std::sync::Arc<crate::continuation::Plan> {
    std::sync::Arc::new(crate::continuation::Plan {
        entries: vec![crate::continuation::Entry {
            code: CodeId(code),
            inputs: 0,
            outputs: 1,
            exits: vec![crate::continuation::Exit::Complete].into_boxed_slice(),
        }]
        .into_boxed_slice(),
    })
}

#[test]
fn property_read_cache_handles_mutation_and_heap_identity() {
    let mut cache = PropertyReadCache::default();
    for padding in [0, 5, 1] {
        let heap = Heap::default();
        let prototype = heap.object().unwrap();
        let object = heap.object_with_prototype(prototype).unwrap();
        for index in 0..padding {
            assert!(heap.set_property(prototype, &format!("padding{index}"), Value::TRUE));
        }
        let read = |cache: &mut PropertyReadCache| match heap
            .read_property_cached(object, "value", cache)
        {
            Some(PropertyRead::Value(value)) => value,
            _ => panic!("expected an object property value"),
        };
        assert_eq!(read(&mut cache), Value::UNDEFINED);
        assert!(heap.set_property(prototype, "value", Value::number(7.0)));
        assert_eq!(read(&mut cache), Value::number(7.0));
        assert!(heap.set_property(prototype, "value", Value::number(11.0)));
        assert_eq!(read(&mut cache), Value::number(11.0));
        assert!(heap.set_property(prototype, "0", Value::TRUE));
        assert_eq!(read(&mut cache), Value::number(11.0));
        assert!(heap.set_property(object, "value", Value::FALSE));
        assert_eq!(read(&mut cache), Value::FALSE);
        assert!(heap.set_property(object, "0", Value::TRUE));
        assert_eq!(read(&mut cache), Value::FALSE);
        heap.release(object);
        heap.release(prototype);
        heap.drain(1000);
    }
}

#[test]
fn deletion_reuses_shapes_and_releases_values() {
    let heap = Heap::default();
    let object = heap.object().unwrap();
    let child = heap.object().unwrap();
    let id = heap_id(object).unwrap();
    let root = heap.0.borrow().slot(id).shape;
    heap.set_property(object, "a", Value::number(1.0));
    let first = heap.0.borrow().slot(id).shape;
    for _ in 0..10 {
        heap.set_property(object, "b", child);
        assert_eq!(heap.delete_property(object, "b"), Some(true));
        assert_eq!(heap.0.borrow().slot(id).shape, first);
    }
    heap.set_property(object, "b", child);
    heap.set_property(object, "c", Value::number(3.0));
    heap.release(child);
    heap.delete_property(object, "b");
    heap.drain(100);
    assert!(!heap.valid(child));
    let peer = heap.object().unwrap();
    heap.set_property(peer, "a", Value::number(1.0));
    heap.set_property(peer, "c", Value::number(3.0));
    assert_eq!(
        heap.0.borrow().slot(id).shape,
        heap.0.borrow().slot(heap_id(peer).unwrap()).shape
    );
    heap.delete_property(object, "a");
    heap.delete_property(object, "c");
    assert_eq!(heap.0.borrow().slot(id).shape, root);
    heap.release(object);
    heap.release(peer);
}

#[test]
fn deleting_native_properties_invalidates_guards() {
    let heap = Heap::default();
    let object = heap.object().unwrap();
    let native = heap.native_function(NativeFunction::Object).unwrap();
    heap.set_property(object, "method", native);
    heap.release(native);
    let epoch = heap.0.borrow().epoch;
    assert_eq!(heap.delete_property(object, "method"), Some(true));
    assert_eq!(heap.0.borrow().epoch, epoch + 1);
    assert_eq!(heap.delete_property(object, "method"), Some(true));
    assert_eq!(heap.0.borrow().epoch, epoch + 1);
    heap.release(object);
}

#[test]
fn array_named_properties_share_shapes_and_own_values() {
    let heap = Heap::default();
    let first = heap.array(&[Value::number(1.0)]).unwrap();
    let second = heap.array(&[]).unwrap();
    let child = heap.object().unwrap();
    assert!(heap.set_property(first, "child", child));
    assert!(heap.set_property(second, "child", Value::TRUE));
    let shape = heap.0.borrow().slot(heap_id(first).unwrap()).shape;
    assert_eq!(heap.0.borrow().slot(heap_id(second).unwrap()).shape, shape);
    heap.release(child);
    assert!(heap.valid(child));
    assert_eq!(heap.delete_property(first, "child"), Some(true));
    heap.drain(1000);
    assert!(!heap.valid(child));
    assert!(heap.set_property(first, "child", second));
    assert_eq!(heap.0.borrow().slot(heap_id(first).unwrap()).shape, shape);
    assert_eq!(heap.get(first, 0), Some(Value::number(1.0)));
    assert_eq!(heap.property(first, "child"), Some(second));
    heap.release(second);
    heap.release(first);
    heap.drain(1000);
    assert!(!heap.valid(second));
}
