from utils.pixel import *


def random_pixel(d_cov, d_met, d_yie, n=1):
    result = []
    cid = d_cov.coord_id.to_frame().sample(n, ignore_index=True).coord_id
    for i in range(n):
        result.append(pixel(cid[i], d_cov, d_met, d_yie))
    return result

# def unix_date_seqence(pd_date, step=24*3600):
# # Function which helps with different time-formats
# #    Converts pandas 'dateSeries' to unix time and provides
# #    unix-numpy and pandas series with ´step´-seconds
# #  Output:
# #    'pd_date in unix-numpy array',
# #    'unix-numpy series with `step`-increase',
# #    'pandas-series with `step`-increase'
# #  Default: increase of one day
#     # convert to unix
#     x = pd.to_datetime(pd_date).astype(int) / 10**9
#     x = x.to_numpy()
#     # get equaliy spaced dates
#     xs_np = np.arange(x.min(), x.max(), step)  # each day
#     # convert from unix to %Y-%m-%d
#     xs_pd = pd.DataFrame(xs_np * 10**9)
#     xs_pd = pd.to_datetime(xs_pd[0], format="%Y-%m-%d")
#     return x, xs_np, xs_pd
