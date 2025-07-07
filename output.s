    .intel_syntax noprefix
    .globl main
main:
    push rbp
    mov rbp, rsp
    .section .rodata
.LC0:
    .string "hello, world\\n"
    .text
    lea rax, .LC0
    push rax
    call printf
    mov rax, 0
    push rax
    mov rsp, rbp
    pop rbp
    ret
