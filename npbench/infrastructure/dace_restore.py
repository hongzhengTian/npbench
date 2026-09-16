"""Restore DaCe binaries while recording narrowly bounded metadata adaptation."""
import copy
import hashlib
import json
from pathlib import Path


def normalize_nested_zero_offsets(document):
    """Fix only rank-mismatched zero offsets in nested Array descriptors.

    A zero offset contributes zero to address calculation at either rank.
    Top-level ABI, shape, strides, nonzero offsets and executable bytes are
    never changed. All other deserialization errors remain errors.
    """
    normalized = copy.deepcopy(document)
    changes = []

    def visit(value, path, graph_depth=-1):
        if isinstance(value, dict):
            depth = graph_depth + int(value.get('type') == 'SDFG')
            if value.get('type') == 'Array' and depth >= 1:
                attrs = value.get('attributes', {})
                shape, offset, strides = (attrs.get(k) for k in ('shape', 'offset', 'strides'))
                if (isinstance(shape, list) and shape and isinstance(offset, list) and offset
                        and len(offset) != len(shape) and isinstance(strides, list) and len(strides) == len(shape)
                        and all(type(x) in (int, str) and x in (0, '0') for x in offset)):
                    replacement = ['0'] * len(shape)
                    changes.append({'path': path + '/attributes/offset', 'before': offset, 'after': replacement})
                    attrs['offset'] = replacement
            for key, child in value.items():
                visit(child, path + '/' + str(key), depth)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                visit(child, path + '/' + str(i), graph_depth)
    visit(normalized, '')
    return normalized, changes


def load_artifact(folder, *, audit_path=None):
    """No recompilation and no mutation of the original SDFG or library."""
    import dace
    from dace.sdfg.utils import load_precompiled_sdfg
    folder = Path(folder)
    audit = {'schema': 1, 'adapter': 'official_unmodified', 'changes': []}
    with dace.config.set_temporary('testing', 'deserialize_exception', value=True):
        try:
            compiled = load_precompiled_sdfg(str(folder))
        except TypeError as error:
            if str(error) != 'Offset must be the same size as shape':
                raise
            original = (folder / 'program.sdfg').read_bytes()
            document, changes = normalize_nested_zero_offsets(json.loads(original))
            if not changes:
                raise
            sdfg = dace.SDFG.from_json(document)
            # Same runtime constructors used by DaCe's official loader.
            from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL
            suffix = dace.Config.get('compiler', 'library_extension')
            library = folder / 'build' / ('lib' + sdfg.name + '.' + suffix)
            compiled = CompiledSDFG(sdfg, ReloadableDLL(str(library), sdfg.name))
            audit.update(adapter='nested_zero_offset_rank', original_error=str(error), changes=changes,
                         original_sdfg_sha256=hashlib.sha256(original).hexdigest())
    if audit_path is not None:
        Path(audit_path).write_text(json.dumps(audit, indent=2) + '\n')
    return compiled
