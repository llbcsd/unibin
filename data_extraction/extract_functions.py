from elftools.elf.elffile import ELFFile
import time
import os


def get_architecture(elf):
    header = elf.header
    e_machine = header.e_machine

    if e_machine == 'EM_386':
        return 'x86'  # 32-bit
    elif e_machine == 'EM_X86_64':
        return 'x86'  # 64-bit
    elif e_machine == 'EM_ARM':
        return 'arm'
    elif e_machine == 'EM_MIPS':
        return 'mips'
    else:
        return None
    

def get_function_addr(elf_filename, function_name):
    with open(elf_filename, 'rb') as file:
        elf = ELFFile(file)

        symtab_section = elf.get_section_by_name('.symtab')

        if symtab_section is None:
            print("Symbol table section not found.")
            return None

        for symbol in symtab_section.iter_symbols():
            if symbol.name == function_name:
                function_address = symbol['st_value']
                function_size = symbol['st_size']

                return function_address, function_size

        print(f"Function '{function_name}' not found in the symbol table.")
        return None


def get_dynsym_func_list(dyn_sym):
    functions = []
    if dyn_sym is None:
        return []
    for sym in dyn_sym.iter_symbols():
        if sym.entry.st_info['type'] == 'STT_FUNC' and sym.entry['st_shndx'] == 'SHN_UNDEF':
            func_name = sym.name
            if func_name not in functions:
                functions.append(func_name)
    return functions


def get_all_function_addr(elf):
    all_func_dict = {}

    symtab_section = elf.get_section_by_name('.symtab')

    if symtab_section is None:
        print("Symbol table section not found.")
        return None

    dyn_funcs = get_dynsym_func_list(elf.get_section_by_name('.dynsym'))

    for symbol in symtab_section.iter_symbols():
        if symbol['st_info']['type'] == 'STT_FUNC' and symbol.name not in dyn_funcs and symbol['st_size'] != 0:
            all_func_dict[symbol.name] = {
                'addr': symbol['st_value'],
                'size': symbol['st_size']
            }

    return all_func_dict


def get_function_bytes_from_address(elf, file, function_address, func_size):
    for section in elf.iter_sections():
        if section['sh_addr'] <= function_address < section['sh_addr'] + section['sh_size']:
            offset = function_address - section['sh_addr'] + section['sh_offset']
            file.seek(offset)
            function_bytes = file.read(func_size)
            return function_bytes


def get_text_section(elf):
    for section in elf.iter_sections():
        if section.name == '.text':
            sh_addr = section['sh_addr']
            sh_offset = section['sh_offset']
            sh_size = section['sh_size']
            return sh_addr, sh_offset, sh_size
    return None, None


def get_all_function_hex(elf, text_addr, text_offset, text_size, file):
    all_func_dict = {}

    symtab_section = elf.get_section_by_name('.symtab')

    if symtab_section is None:
        print("Symbol table section not found.")
        return None

    dyn_funcs = get_dynsym_func_list(elf.get_section_by_name('.dynsym'))

    for symbol in symtab_section.iter_symbols():
        if symbol['st_info']['type'] == 'STT_FUNC' and symbol.name not in dyn_funcs and symbol['st_size'] != 0:
            func_name = symbol.name
            func_addr = symbol['st_value']
            func_size = symbol['st_size']
            if text_addr <= func_addr < text_addr + text_size:
                offset = func_addr - text_addr + text_offset
                file.seek(offset)
                function_bytes = file.read(func_size)
                save_bytes = " ".join(f"{byte:02x}" for byte in function_bytes)
                all_func_dict[func_name] = {
                    'hex': save_bytes
                }

    return all_func_dict


def getTarget(path, prefixfilter=None):
    target = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if prefixfilter is None:
                target.append(os.path.join(root, file))
            else:
                for prefix in prefixfilter:
                    if file.startswith(prefix):
                        target.append(os.path.join(root, file))
    return target


def extract_functions_from_binary(file_path):
    file = open(file_path, 'rb')
    elf = ELFFile(file)

    arch = get_architecture(elf)
    if arch is None:
        print('[!] unkown ISA')
        exit(0)

    text_addr, text_offset, text_size = get_text_section(elf)
    func_dict = get_all_function_hex(elf, text_addr, text_offset, text_size, file)
    file.close()

    print('[-] extract functions: ', len(func_dict))

    return func_dict, arch


def extract_functions_from_dir(dir_path):
    all_files = getTarget(dir_path)

    res_dict = {}

    for file_path in all_files:
        func_dict, arch = extract_functions_from_binary(file_path)
        res_dict[file_path] = {
            'arch': arch,
            'func_dict': func_dict,
        }
    return res_dict


if __name__ == '__main__':
    start = time.time()

    all_files = getTarget('/home/liu/project/rtime_elf')

    for file_path in all_files:

        file = open(file_path, 'rb')

        elf = ELFFile(file)

        text_addr, text_offset, text_size = get_text_section(elf)

        func_dict = get_all_function_hex(elf, text_addr, text_offset, text_size, file)

        file.close()

        print(len(func_dict))

    end = time.time()
    print(f"[*] Time Cost: {end - start} seconds")

