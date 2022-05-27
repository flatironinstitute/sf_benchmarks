#!/usr/bin/env python3

import toml
from typing import List
from pprint import pformat
import itertools

def get_veclevel(stype, iset):
    valsizes = {'float': 4*8, 'double': 8*8}
    regsizes = { 'avx2': 256, 'avx512': 512}

    if iset == 'x86_64':
        veclevel = 1
    elif iset == 'unknown':
        veclevel = 0
    else:
        veclevel = regsizes[iset] // valsizes[stype]

    return veclevel


class Function:
    """hi."""

    def __init__(self, lname, calltemplate, implements, stype, iset, domain,
                 override_template=False):
        """hi."""
        self.lname: str = lname
        self.calltemplate = calltemplate
        self.iset: str = iset
        self.implements: str = implements
        self.stype: str = stype
        self.veclevel = get_veclevel(stype, iset)
        self.domain: List[float] = config.get('domain', [0.0, 1.0])
        self.override_template: bool = override_template

    def __repr__(self):
        """Print this object."""
        return pformat(vars(self))

    def gen_map_elem(self):
        """Generate map, lambda expression pairs in C++ format"""
        if isinstance(self.calltemplate, str):
            calltemplate = self.calltemplate
        else:
            calltemplate = self.calltemplate[self.stype + "_" + self.iset]

        if self.override_template:
            template = """{{{{"{lname}", "{func}", {veclevel} }}, """ + calltemplate + "}}"
        else:
            if self.veclevel == 1:
                template = """{{{{"{lname}", "{func}", {veclevel} }}, scalar_func_map<{stype}>([]({stype} x) -> {stype} {{ return """ + calltemplate + """; }})}}"""
            else:
                template = """{{{{"{lname}", "{func}", {veclevel} }}, vec_func_map<{vectype}, {stype}>([]({vectype} x) -> {vectype} {{ return """ + calltemplate + """; }})}}"""

        vectype = "Vec{}{}".format(self.veclevel, self.stype[0])

        map_elem = template.format(lname=self.lname, func=self.implements, stype=self.stype,
                                   vectype=vectype, veclevel=self.veclevel)
        return map_elem


config = toml.load("funcs.toml")
domains = config.pop('domains')

funcs = {}
fnames = set()
lnames = set()
fdnames = set()

for key in domains.keys():
    fdnames.add(key)

for lname, lconfig in config.items():
    types = lconfig.pop('types')
    instructions = lconfig.pop('instructions')

    lnames.add(lname)

    for fname, calltemplate in lconfig.get('calltemplates', {}).items():
        fnames.add(fname)
        for stype, iset in itertools.product(types, instructions):
            veclevel = get_veclevel(stype, iset)
            mapname = "funs_{}x{}".format(stype[0], veclevel)
            func = Function(lname, calltemplate, fname, stype, iset,
                            domains.get(fname, [0.0, 1.0]))
            mapkey = "funcs_" + stype
            if mapkey not in funcs:
                funcs[mapkey] = [func]
            else:
                funcs[mapkey].append(func)

    for fname, calltemplate in lconfig.get('overrides', {}).items():
        fnames.add(fname)
        for stype, iset in itertools.product(types, instructions):
            veclevel = get_veclevel(stype, iset)
            mapname = "funs_{}x{}".format(stype[0], veclevel)
            func = Function(lname, calltemplate, fname, stype, iset,
                            domains.get(fname, [0.0, 1.0]),
                            override_template=True)
            mapkey = "funcs_" + stype
            if mapkey not in funcs:
                funcs[mapkey] = [func]
            else:
                funcs[mapkey].append(func)

print("#include <sf_benchmarks.hpp>")
print("#include <sf_libraries.hpp>")
print("namespace sf::functions {")

for mapname, funcl in funcs.items():
    print("std::unordered_map<function_key, multi_eval_func<{}>> {} = {{".format(funcl[0].stype, mapname))
    print(",\n".join([func.gen_map_elem() for func in funcl]))
    print("};\n")

print("std::unordered_map<function_key, multi_eval_func<float>> &get_float_funs() { return funcs_float; }")
print("std::unordered_map<function_key, multi_eval_func<double>> &get_double_funs() { return funcs_double; }")

print("}")

# print(lnames)
# print(fnames)
# print(fnames.difference(fdnames))
