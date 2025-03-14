	.text
	.file	"temp.cpp"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function main
.LCPI0_0:
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	3                               # 0x3
.LCPI0_1:
	.long	4                               # 0x4
	.long	4                               # 0x4
	.long	4                               # 0x4
	.long	4                               # 0x4
.LCPI0_2:
	.long	8                               # 0x8
	.long	8                               # 0x8
	.long	8                               # 0x8
	.long	8                               # 0x8
.LCPI0_3:
	.long	12                              # 0xc
	.long	12                              # 0xc
	.long	12                              # 0xc
	.long	12                              # 0xc
.LCPI0_4:
	.long	16                              # 0x10
	.long	16                              # 0x10
	.long	16                              # 0x10
	.long	16                              # 0x10
.LCPI0_5:
	.long	20                              # 0x14
	.long	20                              # 0x14
	.long	20                              # 0x14
	.long	20                              # 0x14
.LCPI0_6:
	.long	24                              # 0x18
	.long	24                              # 0x18
	.long	24                              # 0x18
	.long	24                              # 0x18
.LCPI0_7:
	.long	28                              # 0x1c
	.long	28                              # 0x1c
	.long	28                              # 0x1c
	.long	28                              # 0x1c
.LCPI0_8:
	.long	32                              # 0x20
	.long	32                              # 0x20
	.long	32                              # 0x20
	.long	32                              # 0x20
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$40, %rsp
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	leaq	-72(%rbp), %rdi
	movl	$64, %esi
	movl	$8192, %edx                     # imm = 0x2000
	callq	posix_memalign@PLT
	testl	%eax, %eax
	jne	.LBB0_11
# %bb.1:
	movq	-72(%rbp), %r12
	movl	$1024, %edi                     # imm = 0x400
	callq	malloc@PLT
	movq	%rax, %rbx
	movdqa	.LCPI0_0(%rip), %xmm0           # xmm0 = [0,1,2,3]
	xorl	%eax, %eax
	movdqa	.LCPI0_1(%rip), %xmm1           # xmm1 = [4,4,4,4]
	movdqa	.LCPI0_2(%rip), %xmm2           # xmm2 = [8,8,8,8]
	movdqa	.LCPI0_3(%rip), %xmm3           # xmm3 = [12,12,12,12]
	movdqa	.LCPI0_4(%rip), %xmm4           # xmm4 = [16,16,16,16]
	movdqa	.LCPI0_5(%rip), %xmm5           # xmm5 = [20,20,20,20]
	movdqa	.LCPI0_6(%rip), %xmm6           # xmm6 = [24,24,24,24]
	movdqa	.LCPI0_7(%rip), %xmm7           # xmm7 = [28,28,28,28]
	movdqa	.LCPI0_8(%rip), %xmm8           # xmm8 = [32,32,32,32]
	.p2align	4, 0x90
.LBB0_2:                                # =>This Inner Loop Header: Depth=1
	movdqa	%xmm0, %xmm9
	paddd	%xmm1, %xmm9
	movdqu	%xmm0, (%rbx,%rax,4)
	movdqu	%xmm9, 16(%rbx,%rax,4)
	movdqa	%xmm0, %xmm9
	paddd	%xmm2, %xmm9
	movdqa	%xmm0, %xmm10
	paddd	%xmm3, %xmm10
	movdqu	%xmm9, 32(%rbx,%rax,4)
	movdqu	%xmm10, 48(%rbx,%rax,4)
	movdqa	%xmm0, %xmm9
	paddd	%xmm4, %xmm9
	movdqa	%xmm0, %xmm10
	paddd	%xmm5, %xmm10
	movdqu	%xmm9, 64(%rbx,%rax,4)
	movdqu	%xmm10, 80(%rbx,%rax,4)
	movdqa	%xmm0, %xmm9
	paddd	%xmm6, %xmm9
	movdqa	%xmm0, %xmm10
	paddd	%xmm7, %xmm10
	movdqu	%xmm9, 96(%rbx,%rax,4)
	movdqu	%xmm10, 112(%rbx,%rax,4)
	addq	$32, %rax
	paddd	%xmm8, %xmm0
	cmpq	$256, %rax                      # imm = 0x100
	jne	.LBB0_2
# %bb.3:
	movq	$1, -64(%rbp)
	movq	$0, -56(%rbp)
	movq	$1, -48(%rbp)
	leaq	-64(%rbp), %rsi
	leaq	-56(%rbp), %r15
	movq	%r15, %rdi
	movq	%r15, %rdx
	callq	_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE
	movl	4(%rbx), %ecx
	movl	(%rbx,%rax,4), %edx
	movl	%edx, 4(%rbx)
	movl	%ecx, (%rbx,%rax,4)
	movl	$4, %r13d
	movl	$1016, %r14d                    # imm = 0x3F8
	jmp	.LBB0_4
	.p2align	4, 0x90
.LBB0_6:                                #   in Loop: Header=BB0_4 Depth=1
	xorl	%edx, %edx
	divq	%r13
.LBB0_7:                                #   in Loop: Header=BB0_4 Depth=1
	movl	-8(%rbx,%r13,4), %ecx
	movl	(%rbx,%rax,4), %esi
	movl	%esi, -8(%rbx,%r13,4)
	movl	%ecx, (%rbx,%rax,4)
	movl	-4(%rbx,%r13,4), %eax
	movl	(%rbx,%rdx,4), %ecx
	movl	%ecx, -4(%rbx,%r13,4)
	movl	%eax, (%rbx,%rdx,4)
	addq	$2, %r13
	addq	$-8, %r14
	je	.LBB0_8
.LBB0_4:                                # =>This Inner Loop Header: Depth=1
	leaq	-1(%r13), %rax
	imulq	%r13, %rax
	decq	%rax
	movq	$0, -56(%rbp)
	movq	%rax, -48(%rbp)
	movq	%r15, %rdi
	leaq	-64(%rbp), %rsi
	movq	%r15, %rdx
	callq	_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE
	movq	%rax, %rcx
	orq	%r13, %rcx
	shrq	$32, %rcx
	jne	.LBB0_6
# %bb.5:                                #   in Loop: Header=BB0_4 Depth=1
                                        # kill: def $eax killed $eax killed $rax
	xorl	%edx, %edx
	divl	%r13d
                                        # kill: def $edx killed $edx def $rdx
                                        # kill: def $eax killed $eax def $rax
	jmp	.LBB0_7
.LBB0_8:
	movl	$0, -56(%rbp)
	#APP
	# LLVM-MCA-BEGIN
	#NO_APP
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_9:                                # =>This Inner Loop Header: Depth=1
	movslq	(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	4(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	8(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	12(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	16(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	20(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	24(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	28(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	32(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	36(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	40(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	44(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	48(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	52(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	56(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	60(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	64(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	68(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	72(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	76(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	80(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	84(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	88(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	92(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	96(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	100(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	104(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	108(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	112(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	116(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	120(%rbx,%rax), %rcx
	shlq	$5, %rcx
	movaps	(%r12,%rcx), %xmm0
	movaps	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	movslq	124(%rbx,%rax), %rcx
	shlq	$5, %rcx
	subq	$-128, %rax
	movaps	(%r12,%rcx), %xmm0
	movdqa	16(%r12,%rcx), %xmm0
	movl	$1, -56(%rbp)
	cmpq	$1024, %rax                     # imm = 0x400
	jne	.LBB0_9
# %bb.10:
	#APP
	# LLVM-MCA-END
	#NO_APP
	movq	-72(%rbp), %rdi
	callq	free@PLT
	xorl	%eax, %eax
	addq	$40, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB0_11:
	.cfi_def_cfa %rbp, 16
	callq	abort@PLT
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.section	.text._ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE,"axG",@progbits,_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE,comdat
	.weak	_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE # -- Begin function _ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE
	.p2align	4, 0x90
	.type	_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE,@function
_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE: # @_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$24, %rsp
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	movq	%rsi, %rbx
	movq	(%rdx), %rcx
	movq	8(%rdx), %r13
	subq	%rcx, %r13
	cmpq	$2147483644, %r13               # imm = 0x7FFFFFFC
	ja	.LBB1_6
# %bb.1:
	incq	%r13
	movl	$2147483645, %eax               # imm = 0x7FFFFFFD
	xorl	%edx, %edx
	divl	%r13d
	movl	%eax, %esi
	imulq	%rsi, %r13
	movq	(%rbx), %rdx
	movabsq	$8589934597, %r8                # imm = 0x200000005
	.p2align	4, 0x90
.LBB1_2:                                # =>This Inner Loop Header: Depth=1
	imulq	$16807, %rdx, %rdi              # imm = 0x41A7
	movq	%rdi, %rax
	mulq	%r8
	movq	%rdi, %rax
	subq	%rdx, %rax
	shrq	%rax
	addq	%rdx, %rax
	shrq	$30, %rax
	movq	%rax, %rdx
	shlq	$31, %rdx
	subq	%rdx, %rax
	leaq	(%rdi,%rax), %rdx
	addq	%rdi, %rax
	decq	%rax
	cmpq	%r13, %rax
	jae	.LBB1_2
# %bb.3:
	movq	%rdx, (%rbx)
	movq	%rax, %rdx
	shrq	$32, %rdx
	je	.LBB1_4
# %bb.5:
	xorl	%edx, %edx
	divq	%rsi
	jmp	.LBB1_10
.LBB1_6:
	cmpq	$2147483645, %r13               # imm = 0x7FFFFFFD
	jne	.LBB1_7
# %bb.11:
	imulq	$16807, (%rbx), %rsi            # imm = 0x41A7
	movabsq	$8589934597, %rdx               # imm = 0x200000005
	movq	%rsi, %rax
	mulq	%rdx
	movq	%rsi, %rax
	subq	%rdx, %rax
	shrq	%rax
	addq	%rdx, %rax
	shrq	$30, %rax
	movq	%rax, %rdx
	shlq	$31, %rdx
	subq	%rdx, %rax
	leaq	(%rsi,%rax), %rdx
	movq	%rdx, (%rbx)
	addq	%rsi, %rax
	decq	%rax
	jmp	.LBB1_10
.LBB1_7:
	movq	%rdi, %r14
	movq	%rdx, -48(%rbp)                 # 8-byte Spill
	movq	%r13, %rax
	shrq	%rax
	movabsq	$-9223372028264841207, %rcx     # imm = 0x8000000200000009
	mulq	%rcx
	movq	%rdx, %r15
	shrq	$29, %r15
	movabsq	$8589934597, %r12               # imm = 0x200000005
	.p2align	4, 0x90
.LBB1_8:                                # =>This Inner Loop Header: Depth=1
	movq	$0, -64(%rbp)
	movq	%r15, -56(%rbp)
	movq	%r14, %rdi
	movq	%rbx, %rsi
	leaq	-64(%rbp), %rdx
	callq	_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE
	movq	%rax, %rcx
	addq	%rax, %rax
	shlq	$31, %rcx
	subq	%rax, %rcx
	imulq	$16807, (%rbx), %rsi            # imm = 0x41A7
	movq	%rsi, %rax
	mulq	%r12
	movq	%rsi, %rdi
	subq	%rdx, %rdi
	shrq	%rdi
	addq	%rdx, %rdi
	shrq	$30, %rdi
	movq	%rdi, %rax
	shlq	$31, %rax
	subq	%rax, %rdi
	addq	%rsi, %rdi
	leaq	(%rcx,%rdi), %rax
	decq	%rax
	cmpq	%r13, %rax
	seta	%dl
	movq	%rdi, (%rbx)
	cmpq	%rcx, %rax
	setb	%cl
	orb	%dl, %cl
	jne	.LBB1_8
# %bb.9:
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rcx
	jmp	.LBB1_10
.LBB1_4:
                                        # kill: def $eax killed $eax killed $rax
	xorl	%edx, %edx
	divl	%esi
                                        # kill: def $eax killed $eax def $rax
.LBB1_10:
	addq	%rax, %rcx
	movq	%rcx, %rax
	addq	$24, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end1:
	.size	_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE, .Lfunc_end1-_ZNSt24uniform_int_distributionImEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEmRT_RKNS0_10param_typeE
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 19.1.1 (1ubuntu1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
