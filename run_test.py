from method import main  # replace with your filename

# SIMP
main(
    nelx=60,  # number of elements in x-direction
    nely=30,  # number of elements in y-direction
    volfrac=0.5,  # target volume fraction
    penal=3.0,  # penalization factor
    rmin=1.5,  # filter radius
    ft=1,  # filter type: 1=density filter, 0=sensitivity filter
    xsolv=0,  # optimizer: 0=OC, 1=MMA
    method="SIMP",  # choose SIMP method
)


# RAMP:
main(
    nelx=60,
    nely=30,
    volfrac=0.5,
    penal=3.0,
    rmin=1.5,
    ft=1,
    xsolv=0,
    method="RAMP",
    ramp_q=50,
)


# BESO
main(
    nelx=60,
    nely=30,
    volfrac=0.5,
    penal=3.0,
    rmin=1.5,
    ft=1,
    xsolv=0,
    method="BESO",
    b_erase=0.02,
)
