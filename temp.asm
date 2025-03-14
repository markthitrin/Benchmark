	.file	"temp.cpp"
	.text
#APP
	.globl _ZSt21ios_base_library_initv
#NO_APP
	.section	.text._ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE.isra.0,"axG",@progbits,_ZSt7shuffleIPiSt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEvT_S3_OT0_,comdat
	.align 2
	.p2align 4
	.type	_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE.isra.0, @function
_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE.isra.0:
.LFB11077:
	.cfi_startproc
	subq	%rsi, %rdx
	pushq	%r14
	.cfi_def_cfa_offset 16
	.cfi_offset 14, -16
	movq	%rdx, %r14
	pushq	%r13
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
	movq	%rdi, %r13
	pushq	%r12
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	movq	%rsi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
	cmpq	$2147483644, %rdx
	ja	.L2
	leaq	1(%rdx), %rdi
	movl	$2147483645, %eax
	xorl	%edx, %edx
	movabsq	$8589934597, %r8
	divq	%rdi
	movq	0(%r13), %rdx
	imulq	%rax, %rdi
	movq	%rax, %r9
	.p2align 4,,10
	.p2align 3
.L3:
	imulq	$16807, %rdx, %rsi
	movq	%rsi, %rax
	movq	%rsi, %rcx
	mulq	%r8
	subq	%rdx, %rcx
	shrq	%rcx
	addq	%rcx, %rdx
	shrq	$30, %rdx
	movq	%rdx, %rcx
	salq	$31, %rcx
	subq	%rdx, %rcx
	subq	%rcx, %rsi
	leaq	-1(%rsi), %rax
	movq	%rsi, %rdx
	cmpq	%rdi, %rax
	jnb	.L3
	xorl	%edx, %edx
	movq	%rsi, 0(%r13)
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	divq	%r9
	popq	%rbp
	.cfi_def_cfa_offset 32
	addq	%r12, %rax
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L2:
	.cfi_restore_state
	cmpq	$2147483645, %rdx
	je	.L5
	movabsq	$-9223372028264841207, %rax
	shrq	%rdx
	movabsq	$8589934597, %rbp
	mulq	%rdx
	movq	%rdx, %rbx
	shrq	$29, %rbx
.L10:
	xorl	%esi, %esi
	movq	%rbx, %rdx
	movq	%r13, %rdi
	call	_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE.isra.0
	imulq	$16807, 0(%r13), %rsi
	movq	%rax, %rcx
	movq	%rsi, %rax
	mulq	%rbp
	movq	%rsi, %rax
	subq	%rdx, %rax
	shrq	%rax
	addq	%rax, %rdx
	shrq	$30, %rdx
	movq	%rdx, %rax
	salq	$31, %rax
	subq	%rdx, %rax
	subq	%rax, %rsi
	movq	%rcx, %rax
	salq	$30, %rax
	movq	%rsi, %rdx
	movq	%rsi, 0(%r13)
	subq	%rcx, %rax
	subq	$1, %rdx
	addq	%rax, %rax
	addq	%rdx, %rax
	setc	%dl
	movzbl	%dl, %edx
	cmpq	%rax, %r14
	jb	.L10
	testq	%rdx, %rdx
	jne	.L10
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	addq	%r12, %rax
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L5:
	.cfi_restore_state
	imulq	$16807, (%rdi), %rcx
	movabsq	$8589934597, %rax
	mulq	%rcx
	movq	%rcx, %rax
	subq	%rdx, %rax
	shrq	%rax
	addq	%rdx, %rax
	shrq	$30, %rax
	movq	%rax, %rdx
	salq	$31, %rdx
	subq	%rax, %rdx
	movq	%rcx, %rax
	subq	%rdx, %rax
	movq	%rax, (%rdi)
	subq	$1, %rax
	popq	%rbx
	.cfi_def_cfa_offset 40
	addq	%r12, %rax
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE11077:
	.size	_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE.isra.0, .-_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE.isra.0
	.section	.text._ZSt7shuffleIPiSt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEvT_S3_OT0_,"axG",@progbits,_ZSt7shuffleIPiSt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEvT_S3_OT0_,comdat
	.p2align 4
	.weak	_ZSt7shuffleIPiSt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEvT_S3_OT0_
	.type	_ZSt7shuffleIPiSt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEvT_S3_OT0_, @function
_ZSt7shuffleIPiSt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEvT_S3_OT0_:
.LFB10702:
	.cfi_startproc
	endbr64
	cmpq	%rsi, %rdi
	je	.L25
	movq	%rsi, %rcx
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	movl	$2147483645, %eax
	movq	%rsi, %r11
	subq	%rdi, %rcx
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	leaq	4(%rdi), %r10
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	sarq	$2, %rcx
	movq	%rdx, %rbx
	xorl	%edx, %edx
	divq	%rcx
	cmpq	%rcx, %rax
	jb	.L28
	jmp	.L29
	.p2align 4,,10
	.p2align 3
.L18:
	movq	%r10, %rdx
	xorl	%esi, %esi
	movq	%rbx, %rdi
	addq	$4, %r10
	subq	%rbp, %rdx
	sarq	$2, %rdx
	call	_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE.isra.0
	movl	-4(%r10), %edx
	leaq	0(%rbp,%rax,4), %rax
	movl	(%rax), %ecx
	movl	%ecx, -4(%r10)
	movl	%edx, (%rax)
.L28:
	cmpq	%r10, %r11
	jne	.L18
.L23:
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L29:
	.cfi_restore_state
	andl	$1, %ecx
	je	.L30
.L16:
	cmpq	%r11, %r10
	je	.L23
	.p2align 4,,10
	.p2align 3
.L17:
	movq	%r10, %rdx
	xorl	%esi, %esi
	movq	%rbx, %rdi
	addq	$8, %r10
	subq	%rbp, %rdx
	sarq	$2, %rdx
	leaq	2(%rdx), %r12
	addq	$1, %rdx
	imulq	%r12, %rdx
	subq	$1, %rdx
	call	_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE.isra.0
	xorl	%edx, %edx
	movl	-8(%r10), %ecx
	divq	%r12
	leaq	0(%rbp,%rax,4), %rax
	movl	(%rax), %esi
	movl	%esi, -8(%r10)
	movl	%ecx, (%rax)
	leaq	0(%rbp,%rdx,4), %rax
	movl	-4(%r10), %edx
	movl	(%rax), %ecx
	movl	%ecx, -4(%r10)
	movl	%edx, (%rax)
	cmpq	%r10, %r11
	jne	.L17
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L30:
	.cfi_restore_state
	movl	$1, %edx
	xorl	%esi, %esi
	leaq	8(%rbp), %r10
	movq	%rbx, %rdi
	call	_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE.isra.0
	movl	4(%rbp), %edx
	leaq	0(%rbp,%rax,4), %rax
	movl	(%rax), %ecx
	movl	%ecx, 4(%rbp)
	movl	%edx, (%rax)
	jmp	.L16
	.p2align 4,,10
	.p2align 3
.L25:
	.cfi_def_cfa_offset 8
	.cfi_restore 3
	.cfi_restore 6
	.cfi_restore 12
	ret
	.cfi_endproc
.LFE10702:
	.size	_ZSt7shuffleIPiSt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEvT_S3_OT0_, .-_ZSt7shuffleIPiSt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEvT_S3_OT0_
	.section	.text.unlikely,"ax",@progbits
.LCOLDB3:
	.section	.text.startup,"ax",@progbits
.LHOTB3:
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB10383:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movl	$2048, %edx
	movl	$64, %esi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$96, %rsp
	.cfi_offset 13, -24
	.cfi_offset 12, -32
	.cfi_offset 3, -40
	movq	%fs:40, %rax
	movq	%rax, 88(%rsp)
	xorl	%eax, %eax
	leaq	16(%rsp), %rdi
	call	posix_memalign@PLT
	testl	%eax, %eax
	jne	.L38
	vmovdqa	.LC1(%rip), %ymm0
	movl	$256, %edi
	movq	16(%rsp), %r12
	vmovdqa	%ymm0, 32(%rsp)
	vzeroupper
	call	malloc@PLT
	movl	$4, %ecx
	vmovdqa	.LC0(%rip), %xmm0
	vmovd	%ecx, %xmm2
	movq	%rax, %rbx
	leaq	256(%rax), %r13
	vpshufd	$0, %xmm2, %xmm2
	.p2align 4,,10
	.p2align 3
.L33:
	vmovdqa	%xmm0, %xmm1
	addq	$16, %rax
	vpaddd	%xmm2, %xmm0, %xmm0
	vmovdqu	%xmm1, -16(%rax)
	cmpq	%r13, %rax
	jne	.L33
	movq	$1, 24(%rsp)
	leaq	24(%rsp), %rdx
	movq	%r13, %rsi
	movq	%rbx, %rdi
	call	_ZSt7shuffleIPiSt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEvT_S3_OT0_
	movl	$0, 24(%rsp)
#APP
# 31 "temp.cpp" 1
	# LLVM-MCA-BEGIN
# 0 "" 2
#NO_APP
.L34:
	movslq	(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	4(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	8(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	12(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	16(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	20(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	24(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	28(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	32(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	36(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	40(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	44(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	48(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	52(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	56(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	60(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	64(%rbx), %rax
	salq	$5, %rax
	subq	$-128, %rbx
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	-60(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	-56(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	-52(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	-48(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	-44(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	-40(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	-36(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	-32(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	-28(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	-24(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	-20(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	-16(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	-12(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	-8(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	movslq	-4(%rbx), %rax
	salq	$5, %rax
	addq	%r12, %rax
	vmovdqa	(%rax), %ymm0
	cmpq	%rbx, %r13
	jne	.L34
#APP
# 37 "temp.cpp" 1
	# LLVM-MCA-END
# 0 "" 2
#NO_APP
	movq	16(%rsp), %rdi
	vzeroupper
	call	free@PLT
	movq	88(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L40
	leaq	-24(%rbp), %rsp
	xorl	%eax, %eax
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L40:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.type	main.cold, @function
main.cold:
.LFSB10383:
.L38:
	.cfi_def_cfa 6, 16
	.cfi_offset 3, -40
	.cfi_offset 6, -16
	.cfi_offset 12, -32
	.cfi_offset 13, -24
	call	abort@PLT
	.cfi_endproc
.LFE10383:
	.section	.text.startup
	.size	main, .-main
	.section	.text.unlikely
	.size	main.cold, .-main.cold
.LCOLDE3:
	.section	.text.startup
.LHOTE3:
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC0:
	.long	0
	.long	1
	.long	2
	.long	3
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC1:
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.byte	27
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
