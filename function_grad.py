import torch
import time

dtype = torch.float64
def nll_func_avg_batch(X, A, y, std_w):
    X2 = torch.square(X)
    AT = torch.transpose(A, 0, 1)
    yT = torch.transpose(y, 1, 2)
    Sigma_inv = torch.matmul(torch.matmul(A, X2), AT)
    Sigma = torch.inverse(Sigma_inv)
    f_value_1 = torch.logdet(Sigma_inv)
    f_value_2_ = torch.diagonal(torch.matmul(torch.matmul(yT, Sigma), y), dim1=-2, dim2=-1)
    f_value_2 = torch.mean(f_value_2_, dim=-1) / std_w**2
    f_value = f_value_1 + f_value_2
    return f_value

#####################################################################
############# Real/Complex multi-look, A, exact inverse #############
#####################################################################
def nll_grad_mul_block_matrix_batch(x, A, y, x_star, use_complex, diff_A, L_inf, std_w=1.0):
    X = torch.diag_embed(x)  # Create diagonal matrices from x
    X2 = torch.square(X)
    if use_complex:
        if diff_A:
            B_11 = (A @ X2.type(A.dtype) @ A.T.conj()).real
            B_12 = (A.conj() @ X2.type(A.dtype) @ A.T).imag
            B_21 = (A @ X2.type(A.dtype) @ A.T.conj()).imag
            B_22 = (A.conj() @ X2.type(A.dtype) @ A.T).real

            B_inv_11 = torch.inverse(B_11)
            D_CAinvB = torch.inverse(B_22 - B_21 @ B_inv_11 @ B_12)
            Sigma_inv_11 = B_inv_11 + B_inv_11 @ B_12 @ D_CAinvB @ B_21 @ B_inv_11
            Sigma_inv_12 = - B_inv_11 @ B_12 @ D_CAinvB
            Sigma_inv_21 = - D_CAinvB @ B_21 @ B_inv_11
            Sigma_inv_22 = D_CAinvB

            B_inv = 0
        else:
            B_11 = (A @ X2.type(A.dtype) @ A.T.conj()).real

            inv_start = time.time()
            # print('Compute exact matrix inverse')
            B_inv_11 = torch.inverse(B_11)
            inv_end = time.time()
            print('inverse time', inv_end - inv_start)

            Sigma_inv_11 = B_inv_11
            Sigma_inv_12 = 0
            Sigma_inv_21 = 0
            Sigma_inv_22 = B_inv_11
            B_inv = B_inv_11
    else:
        B = torch.matmul(torch.matmul(A.unsqueeze(0), X2), A.transpose(0, 1))
        # print('Compute exact matrix inverse')
        B_inv = torch.inverse(B)
        # B = torch.matmul(torch.matmul(A.unsqueeze(0), X2), A.unsqueeze(0).transpose(1, 2))

    if use_complex:
        if diff_A:
            # grad_vec_1_1_diag = torch.matmul(A.T, torch.matmul((Sigma_inv_11 + Sigma_inv_22).type(A.dtype), A.conj()))
            # grad_vec_1_1 = torch.diagonal(grad_vec_1_1_diag, dim1=-2, dim2=-1).real
            grad_vec_1_1_diag = torch.matmul(A.real.T, torch.matmul((Sigma_inv_11 + Sigma_inv_22), A.real))
            grad_vec_1_1 = torch.diagonal(grad_vec_1_1_diag, dim1=-2, dim2=-1)
            grad_vec_1_2_diag = torch.matmul(A.imag.T, torch.matmul((Sigma_inv_11 + Sigma_inv_22), A.imag))
            grad_vec_1_2 = torch.diagonal(grad_vec_1_2_diag, dim1=-2, dim2=-1)
            grad_vec_1_3_diag = torch.matmul(A.real.T, torch.matmul((Sigma_inv_12 - Sigma_inv_21), A.imag))
            grad_vec_1_3 = torch.diagonal(grad_vec_1_3_diag, dim1=-2, dim2=-1)
            grad_vec_1_4_diag = torch.matmul(A.imag.T, torch.matmul((Sigma_inv_21 - Sigma_inv_12), A.real))
            grad_vec_1_4 = torch.diagonal(grad_vec_1_4_diag, dim1=-2, dim2=-1)
            grad_vec_1 = grad_vec_1_1 + grad_vec_1_2 + grad_vec_1_3 + grad_vec_1_4

            grad_vec_2_1 = - torch.matmul(torch.matmul(A.imag.T, Sigma_inv_11), y.real)
            grad_vec_2_2 = torch.matmul(torch.matmul(A.imag.T, Sigma_inv_21), y.imag)
            grad_vec_2_3 = torch.matmul(torch.matmul(A.real.T, Sigma_inv_21), y.real)
            grad_vec_2_4 = torch.matmul(torch.matmul(A.real.T, Sigma_inv_22), y.imag)
            grad_vec_2_1_sum = torch.square(grad_vec_2_1 + grad_vec_2_2 + grad_vec_2_3 + grad_vec_2_4)
            grad_vec_2_5 = torch.matmul(torch.matmul(A.real.T, Sigma_inv_11), y.real)
            grad_vec_2_6 = torch.matmul(torch.matmul(A.real.T, Sigma_inv_12), y.imag)
            grad_vec_2_7 = torch.matmul(torch.matmul(A.imag.T, Sigma_inv_21), y.real)
            grad_vec_2_8 = torch.matmul(torch.matmul(A.imag.T, Sigma_inv_22), y.imag)
            grad_vec_2_2_sum = torch.square(grad_vec_2_5 + grad_vec_2_6 + grad_vec_2_7 + grad_vec_2_8)
            grad_vec_2 = torch.mean(grad_vec_2_1_sum + grad_vec_2_2_sum, dim=-1)
        else:
            grad_vec_1_1_diag = torch.matmul(A.real.T, torch.matmul((Sigma_inv_11 + Sigma_inv_22), A.real))
            grad_vec_1_1 = torch.diagonal(grad_vec_1_1_diag, dim1=-2, dim2=-1)
            grad_vec_1 = 2 * grad_vec_1_1

            grad_vec_2_1 = torch.matmul(torch.matmul(A.imag.T, Sigma_inv_11), y.real)
            grad_vec_2_4 = - torch.matmul(torch.matmul(A.real.T, Sigma_inv_22), y.imag)
            grad_vec_2_1_sum = torch.square(grad_vec_2_1 + grad_vec_2_4)

            grad_vec_2_5 = torch.matmul(torch.matmul(A.real.T, Sigma_inv_11), y.real)
            grad_vec_2_8 = torch.matmul(torch.matmul(A.imag.T, Sigma_inv_22), y.imag)
            grad_vec_2_2_sum = torch.square(grad_vec_2_5 + grad_vec_2_8)

            grad_vec_2 = torch.mean(grad_vec_2_1_sum + grad_vec_2_2_sum, dim=-1)
    else:
        if L_inf:
            X2_star = torch.square(torch.diag(x_star.squeeze(0)))
            AX2AT = A @ X2_star @ torch.transpose(A, 0, 1)
            ATB_inv = torch.matmul(A.transpose(0, 1), B_inv)
            grad_vec_1 = torch.diagonal(torch.matmul(ATB_inv, A), dim1=-2, dim2=-1)
            grad_vec_2_ = ATB_inv.squeeze(0) @ AX2AT @ ATB_inv.squeeze(0).T
            grad_vec_2 = torch.diagonal(grad_vec_2_)
            grad_vec_2 = grad_vec_2.unsqueeze(0)

        else:
            ATB_inv = torch.matmul(A.transpose(0, 1), B_inv)
            grad_vec_1 = torch.diagonal(torch.matmul(ATB_inv, A), dim1=-2, dim2=-1)
            grad_vec_2_ = torch.square(torch.matmul(ATB_inv, y))
            # print(grad_vec_2_.shape)
            grad_vec_2 = torch.mean(grad_vec_2_, dim=-1)  # average over all measurement looks

    assert grad_vec_1.shape == grad_vec_2.shape == x.shape
    grad_vec = 2 * x * (grad_vec_1 * std_w ** 2 - grad_vec_2)
    # grad_vec = 2 * x * (0.5 * grad_vec_1 * std_w ** 2 - grad_vec_2)
    # print('grad_vec', grad_vec)
    return grad_vec, B_inv

###########################################################################
############### Real/Complex multi-look, A, pseudo inverse ###############
###########################################################################
def nll_grad_mul_block_matrix_pseudo_batch(x, A, y, B_inv, inv_ite, use_Schulz, use_complex, diff_A, device, std_w=1.0):
    # print('use pseudo_inverse_func')
    X = torch.diag_embed(x)  # Create diagonal matrices from x
    X2 = torch.square(X)
    if use_complex:
        if diff_A:
            B_11 = (A @ X2.type(A.dtype) @ A.T.conj()).real
            B_12 = (A.conj() @ X2.type(A.dtype) @ A.T).imag
            B_21 = (A @ X2.type(A.dtype) @ A.T.conj()).imag
            B_22 = (A.conj() @ X2.type(A.dtype) @ A.T).real
            # B_12_T = B_12.permute(0, 2, 1)
            # B_21 = B_12_T
            # print('Compute exact matrix inverse')
            B_inv_11 = torch.inverse(B_11)
            # B_inv_21 = torch.inverse(B_21)
            # B_inv_22 = torch.inverse(B_22)
            # B_inv_12 = torch.inverse(B_12)
            D_CAinvB = torch.inverse(B_22 - B_21 @ B_inv_11 @ B_12)
            Sigma_inv_11 = B_inv_11 + B_inv_11 @ B_12 @ D_CAinvB @ B_21 @ B_inv_11
            Sigma_inv_12 = - B_inv_11 @ B_12 @ D_CAinvB
            Sigma_inv_21 = - D_CAinvB @ B_21 @ B_inv_11
            Sigma_inv_22 = D_CAinvB
            # Sigma_inv_12_T = Sigma_inv_12.permute(0, 2, 1)
            # Sigma_inv_21 = Sigma_inv_12_T
            # AD_BC = B_11 @ Sigma_inv_12 - B_12 @ Sigma_inv_11
            B_inv = 0
        else:
            if use_Schulz:
                print('Approximate matrix inverse with Schulz')
                # approx_error_avg = torch.zeros(inv_ite).type(dtype).to(device)
                B = (A @ X2.type(A.dtype) @ A.T.conj()).real
                NS_start = time.time()
                for iii in range(inv_ite):
                    # B = (A @ X2.type(A.dtype) @ A.T.conj()).real
                    B_inv = 2 * B_inv - B_inv @ B @ B_inv
                    # B_inv = 2 * B_inv - torch.matmul(torch.matmul(B_inv, B), B_inv)

                    # Calculate matrix inverse approximation error
                    # BB_inv = torch.matmul(B, B_inv)
                    # identity = torch.eye(B_inv.size(dim=-1)).type(dtype).to(device).unsqueeze(0)
                    # approx_error = torch.linalg.matrix_norm(torch.sub(identity, BB_inv))
                    # approx_error_avg[iii] = torch.mean(approx_error)
                    # print('inverse approx err:', torch.mean(approx_error).item())
                NS_end = time.time()
                print('Newton-Schulz time', NS_end - NS_start)

            B_inv_11 = B_inv
            Sigma_inv_11 = B_inv_11
            Sigma_inv_12 = 0
            Sigma_inv_21 = 0
            Sigma_inv_22 = B_inv_11
            B_inv = B_inv_11
    else:
        B = torch.matmul(torch.matmul(A.unsqueeze(0), X2), A.transpose(0, 1))
        # print('Compute exact matrix inverse')
        B_inv = torch.inverse(B)
        # B = torch.matmul(torch.matmul(A.unsqueeze(0), X2), A.unsqueeze(0).transpose(1, 2))

    if use_complex:
        if diff_A:
            # grad_vec_1_1_diag = torch.matmul(A.T, torch.matmul((Sigma_inv_11 + Sigma_inv_22).type(A.dtype), A.conj()))
            # grad_vec_1_1 = torch.diagonal(grad_vec_1_1_diag, dim1=-2, dim2=-1).real
            grad_vec_1_1_diag = torch.matmul(A.real.T, torch.matmul((Sigma_inv_11 + Sigma_inv_22), A.real))
            grad_vec_1_1 = torch.diagonal(grad_vec_1_1_diag, dim1=-2, dim2=-1)
            grad_vec_1_2_diag = torch.matmul(A.imag.T, torch.matmul((Sigma_inv_11 + Sigma_inv_22), A.imag))
            grad_vec_1_2 = torch.diagonal(grad_vec_1_2_diag, dim1=-2, dim2=-1)
            grad_vec_1_3_diag = torch.matmul(A.real.T, torch.matmul((Sigma_inv_12 - Sigma_inv_21), A.imag))
            grad_vec_1_3 = torch.diagonal(grad_vec_1_3_diag, dim1=-2, dim2=-1)
            grad_vec_1_4_diag = torch.matmul(A.imag.T, torch.matmul((Sigma_inv_21 - Sigma_inv_12), A.real))
            grad_vec_1_4 = torch.diagonal(grad_vec_1_4_diag, dim1=-2, dim2=-1)
            grad_vec_1 = grad_vec_1_1 + grad_vec_1_2 + grad_vec_1_3 + grad_vec_1_4

            grad_vec_2_1 = - torch.matmul(torch.matmul(A.imag.T, Sigma_inv_11), y.real)
            grad_vec_2_2 = torch.matmul(torch.matmul(A.imag.T, Sigma_inv_21), y.imag)
            grad_vec_2_3 = torch.matmul(torch.matmul(A.real.T, Sigma_inv_21), y.real)
            grad_vec_2_4 = torch.matmul(torch.matmul(A.real.T, Sigma_inv_22), y.imag)
            grad_vec_2_1_sum = torch.square(grad_vec_2_1 + grad_vec_2_2 + grad_vec_2_3 + grad_vec_2_4)
            grad_vec_2_5 = torch.matmul(torch.matmul(A.real.T, Sigma_inv_11), y.real)
            grad_vec_2_6 = torch.matmul(torch.matmul(A.real.T, Sigma_inv_12), y.imag)
            grad_vec_2_7 = torch.matmul(torch.matmul(A.imag.T, Sigma_inv_21), y.real)
            grad_vec_2_8 = torch.matmul(torch.matmul(A.imag.T, Sigma_inv_22), y.imag)
            grad_vec_2_2_sum = torch.square(grad_vec_2_5 + grad_vec_2_6 + grad_vec_2_7 + grad_vec_2_8)
            grad_vec_2 = torch.mean(grad_vec_2_1_sum + grad_vec_2_2_sum, dim=-1)
        else:
            grad_vec_1_1_diag = torch.matmul(A.real.T, torch.matmul((Sigma_inv_11 + Sigma_inv_22), A.real))
            grad_vec_1_1 = torch.diagonal(grad_vec_1_1_diag, dim1=-2, dim2=-1)
            grad_vec_1 = 2 * grad_vec_1_1

            grad_vec_2_1 = torch.matmul(torch.matmul(A.imag.T, Sigma_inv_11), y.real)
            grad_vec_2_4 = - torch.matmul(torch.matmul(A.real.T, Sigma_inv_22), y.imag)
            grad_vec_2_1_sum = torch.square(grad_vec_2_1 + grad_vec_2_4)

            grad_vec_2_5 = torch.matmul(torch.matmul(A.real.T, Sigma_inv_11), y.real)
            grad_vec_2_8 = torch.matmul(torch.matmul(A.imag.T, Sigma_inv_22), y.imag)
            grad_vec_2_2_sum = torch.square(grad_vec_2_5 + grad_vec_2_8)

            grad_vec_2 = torch.mean(grad_vec_2_1_sum + grad_vec_2_2_sum, dim=-1)
    else:
        ATB_inv = torch.matmul(A.transpose(0, 1), B_inv)
        grad_vec_1 = torch.diagonal(torch.matmul(ATB_inv, A), dim1=-2, dim2=-1)
        grad_vec_2_ = torch.square(torch.matmul(ATB_inv, y))
        # print(grad_vec_2_.shape)
        grad_vec_2 = torch.mean(grad_vec_2_, dim=-1)  # average over all measurement looks
    assert grad_vec_1.shape == grad_vec_2.shape == x.shape
    grad_vec = 2 * x * (grad_vec_1 * std_w ** 2 - grad_vec_2)
    # print('grad_vec', grad_vec)
    return grad_vec, B_inv
