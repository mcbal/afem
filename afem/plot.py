# with torch.no_grad():
#     import matplotlib.pyplot as plt

#     def filter_array(a, threshold=1e2):
#         idx = np.where(np.abs(a) > threshold)
#         a[idx] = np.nan
#         return a
#     # Pick a range and resolution for `t`.
#     t_range = torch.arange(0.0, 3.0, 0.001)[:, None]
#     # Calculate function evaluations for every point on grid and plot.
#     out = np.array(self._phi(t_range, h[:1, :, :].repeat(t_range.numel(), 1, 1)).detach().numpy())
#     out_grad = np.array(self._grad_t_phi(
#         t_range, h[:1, :, :].repeat(t_range.numel(), 1, 1)).detach().numpy())
#     f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#     ax1.set_title(f"(Hopefully) Found root of phi'(t) at t = {t[0][0].detach().numpy()}")
#     ax1.plot(t_range.numpy().squeeze(), filter_array(out), 'r-')
#     ax1.axvline(x=t[0].detach().numpy())
#     ax1.set_ylabel("phi(t)")
#     ax2.plot(t_range.numpy().squeeze(), filter_array(out_grad), 'r-')
#     ax2.axvline(x=t[0].detach().numpy())
#     ax2.axhline(y=0.0)
#     ax2.set_xlabel('t')
#     ax2.set_ylabel("phi'(t)")
#     # plt.show()
#     from datetime import datetime
#     plt.savefig(f'{datetime.now()}.png')
