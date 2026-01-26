# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the Linux kernel source tree. The kernel is a monolithic operating system kernel written primarily in C, with growing Rust support. Version: 6.16.0 (NAME = "Baby Opossum Posse").

## Build System

The Linux kernel uses the Kbuild system, which is a recursive Make-based build system with Kconfig for configuration management.

### Essential Build Commands

```bash
# Configuration
make menuconfig              # Interactive menu-based configuration
make defconfig              # Use default configuration for current architecture
make oldconfig              # Update existing .config with new symbols
make localmodconfig         # Configure only for currently loaded modules
make allmodconfig           # Enable all modules
make ARCH=<arch> defconfig  # Cross-compile configuration

# Build
make                        # Build kernel (vmlinux) and modules
make -j$(nproc)            # Parallel build with all CPU cores
make V=1                    # Verbose build showing full commands
make vmlinux               # Build only the bare kernel
make modules               # Build only modules
make bzImage               # Build compressed kernel image (x86/x86_64)

# Installation
make modules_install        # Install modules to /lib/modules/$(uname -r)
make install               # Install kernel (requires configured bootloader)
make headers_install       # Install sanitized kernel headers

# Clean
make clean                 # Remove most generated files, keep config
make mrproper              # Remove all generated files including config
make distclean             # mrproper + remove editor backups and patches
```

### Cross-Compilation

```bash
make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- defconfig
make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- -j$(nproc)
```

Common architectures: x86, x86_64, arm, arm64, riscv, powerpc, s390, mips

### Building Single Components

```bash
make dir/                  # Build all files in directory and below
make dir/file.o           # Build specific object file
make dir/file.ko          # Build specific module with final link
make dir/file.lst         # Build mixed source/assembly listing
make modules_prepare      # Prepare for building external modules
```

## Development Tools

### Code Style and Checking

```bash
# Check code style before submitting patches
scripts/checkpatch.pl <patch-file>
scripts/checkpatch.pl --no-tree -f <source-file>
scripts/checkpatch.pl --strict <patch-file>  # More pedantic checking

# Find maintainers for subsystem
scripts/get_maintainer.pl <patch-file>
scripts/get_maintainer.pl -f <source-file>

# Decode kernel stack traces
scripts/decode_stacktrace.sh vmlinux < dmesg.log
```

### Static Analysis

```bash
make checkstack            # Find stack hogs (functions with large stack usage)
make includecheck         # Check for duplicate included headers
make versioncheck         # Check version.h usage
make coccicheck           # Run Coccinelle semantic patches
make clang-analyzer       # Run Clang static analyzer
make clang-tidy           # Run clang-tidy checks
```

### Code Navigation

```bash
make tags                 # Generate ctags
make TAGS                 # Generate etags
make cscope               # Generate cscope database
make gtags                # Generate GNU GLOBAL tags
```

## Testing

### Kernel Selftests

```bash
# Build and run all selftests (requires root)
make kselftest

# Build selftests without running
make kselftest-all

# Build and install selftests
make kselftest-install INSTALL_KSELFTEST_PATH=/path

# Clean selftest artifacts
make kselftest-clean

# Merge selftest config dependencies
make kselftest-merge
```

Individual selftests are in `tools/testing/selftests/`. Run specific tests:

```bash
cd tools/testing/selftests/<subsystem>
make
./test_script.sh
```

## Rust Support

The kernel has optional Rust support for writing kernel modules and subsystems.

### Rust Commands

```bash
# Check if Rust toolchain is available
make rustavailable

# Format all Rust code
make rustfmt

# Check Rust formatting
make rustfmtcheck

# Generate Rust documentation
make rustdoc

# Run Rust tests
make rusttest
```

### Requirements
- Rust: >= 1.78.0
- bindgen: >= 0.65.1
- See `Documentation/rust/quick-start.rst` for detailed setup

## Documentation

```bash
# Build documentation
make htmldocs             # Build HTML documentation
make pdfdocs              # Build PDF documentation
make cleandocs            # Clean documentation build artifacts
```

Documentation is in `Documentation/` with subdirectories:
- `process/` - Development process, coding style, patch submission
- `admin-guide/` - System administration guide
- `driver-api/` - Driver API reference
- `kbuild/` - Build system documentation
- `rust/` - Rust support documentation

## Architecture

### Directory Structure

- `arch/` - Architecture-specific code (x86, arm64, riscv, etc.)
- `kernel/` - Core kernel code (scheduler, locking, time, signals, etc.)
- `mm/` - Memory management
- `fs/` - Filesystems (VFS and specific filesystems)
- `drivers/` - Device drivers organized by subsystem
- `net/` - Networking stack
- `block/` - Block layer
- `crypto/` - Cryptographic API
- `security/` - Security modules (SELinux, AppArmor, etc.)
- `lib/` - Kernel library functions
- `init/` - Kernel initialization
- `ipc/` - Inter-process communication
- `io_uring/` - Asynchronous I/O framework
- `rust/` - Rust infrastructure and abstractions
- `scripts/` - Build scripts and development tools
- `tools/` - User-space tools for kernel development
- `samples/` - Example code for kernel features
- `include/` - Kernel headers
  - `include/linux/` - Core kernel headers
  - `include/uapi/` - User-space API headers

### Key Subsystems

**Process Management**: Core scheduler in `kernel/sched/`, process lifecycle in `kernel/fork.c`, `kernel/exit.c`

**Memory Management**: Page allocator, slab allocators, virtual memory in `mm/`. Each architecture implements page tables in `arch/*/mm/`

**Virtual File System (VFS)**: Abstraction layer in `fs/` with `fs/*.c` providing core VFS operations. Individual filesystems in `fs/ext4/`, `fs/btrfs/`, etc.

**Networking**: Protocol-independent framework in `net/core/`, protocol implementations in `net/ipv4/`, `net/ipv6/`, etc. Socket layer in `net/socket.c`

**Device Model**: Core in `drivers/base/`, provides bus/driver/device abstraction. Each driver type has a subsystem directory under `drivers/`

**Locking**: Primitives in `kernel/locking/`. Read `Documentation/locking/` before modifying locking code.

## Development Workflow

### Coding Style
- 8-space tabs for indentation (not spaces)
- 80-column line limit (flexible for readability)
- Opening braces on same line for functions, structs
- Always run `scripts/checkpatch.pl` before submitting
- Read `Documentation/process/coding-style.rst`

### Patch Format
- One logical change per patch
- Descriptive subject line (max 50 chars preferred)
- Detailed commit message explaining *why* not *what*
- Sign-off required: `Signed-off-by: Your Name <email@example.com>`
- Use `git format-patch` to generate patches
- Read `Documentation/process/submitting-patches.rst`

### Configuration System (Kconfig)
- Each directory with configurable options has `Kconfig`
- Symbols defined with `config SYMBOL_NAME`
- Dependencies expressed with `depends on` and `select`
- Main entry point is top-level `Kconfig`
- Read `Documentation/kbuild/kconfig-language.rst`

### Makefiles
- Each directory has a `Makefile` specifying build rules
- `obj-y` for built-in objects
- `obj-m` for modules
- `obj-$(CONFIG_SYMBOL)` for conditional compilation
- Read `Documentation/kbuild/makefiles.rst`

## Important Notes

### Cross-Subsystem Changes
Changes affecting multiple subsystems usually need separate patches per maintainer, unless closely related. Use `scripts/get_maintainer.pl` to find the right people.

### Breaking UAPI
Never break user-space API (UAPI) headers in `include/uapi/`. These are part of the kernel-user contract.

### Performance Implications
The kernel is performance-critical. Be aware of hot paths (scheduler, memory allocator, networking fast path). Use appropriate primitives for each context (atomic, RCU, spinlocks, mutexes).

### Context Awareness
- `in_interrupt()` - interrupt context, no sleeping
- `in_atomic()` - atomic context, no sleeping
- Process context - can sleep, use mutexes
- Always consider which context your code runs in

### Memory Allocation
- `kmalloc/kfree` - general purpose, physically contiguous
- `vmalloc/vfree` - large allocations, virtually contiguous
- `GFP_KERNEL` - can sleep
- `GFP_ATOMIC` - cannot sleep, use in interrupt context
- `GFP_USER` - user-space allocation

### Debugging
- `pr_debug()`, `pr_info()`, `pr_err()` for printk debugging
- `CONFIG_DEBUG_*` options in kernel config
- `dmesg` or `/var/log/kern.log` for kernel messages
- `scripts/decode_stacktrace.sh` for decoding oops/panic traces
- `CONFIG_KGDB` for kernel debugging with GDB

### Module Development
Modules allow extending kernel without recompilation:
```bash
# Load module
insmod module.ko
modprobe module_name

# Unload module
rmmod module_name

# List loaded modules
lsmod

# Module information
modinfo module.ko
```

## Building with LLVM/Clang

```bash
make LLVM=1                          # Use Clang and LLVM tools
make LLVM=1 CC=clang-15              # Specific version
make LLVM_IAS=1                      # Use LLVM integrated assembler
```

See `Documentation/kbuild/llvm.rst` for details.

## Common Development Scenarios

### Adding a New Driver
1. Create driver in appropriate `drivers/<subsystem>/` directory
2. Add `Kconfig` entry for configuration
3. Update `Makefile` with `obj-$(CONFIG_YOUR_DRIVER) += yourdriver.o`
4. Test with `make drivers/subsystem/yourdriver.ko`

### Adding a System Call
1. Add entry to architecture's syscall table (`arch/*/entry/syscalls/syscall_*.tbl`)
2. Implement in appropriate subsystem
3. Add to `include/uapi/asm-generic/unistd.h`
4. Read `Documentation/process/adding-syscalls.rst`

### Modifying Core Subsystems
Core subsystems (scheduler, memory management, VFS) require extra scrutiny. Changes must be benchmarked and tested extensively. Discuss on LKML before significant changes.
