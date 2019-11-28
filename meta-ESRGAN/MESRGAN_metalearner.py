#empirical loss

'''
StellEdge:
meta-SGD model

output (theta,alpha)  get loss-Ttest(theta)


θ represents the state of a learner that can be used to initialize the learner for
any new task, and α is a vector of the same size as θ 

while not None do:
    for each batch T contains (x,y) pairs:
        Ltrain(Ti)(θ) ← mean(loss(fθ(x), y))
        θi' ← θ − α ◦∇Ltrain(Ti)(θ)
        Ltest(Ti)(θi') ← mean(loss(fθi'(x), y))
    (θ, α) ← (θ, α) − β∇(θ,α)sigma-Ti(Ltest(Ti)(θi'))

θ and α are (meta-)parameters of the meta-learner to be learned, and ◦ denotes element-wise
product.

'''
