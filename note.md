# 强化学习经典算法实现

## value-base价值学习

### TD（时序差分）算法

蒙特卡洛方法对价值函数的增量更新方式（$\alpha$表示更新步长） ：
$$
U_t \gets U_t + \alpha [G_t - U_t]
$$

时序差分算法用当前获得的奖励，加上下一个状态的价值估计来作为在当前状态会获得的回报，不需要等到结束就能估计出结果（$\gamma$为折现率）。
$$
\begin{align}
& U_t = R_t + \gamma \cdot U_{t + 1} \\
& y_t = R_t + \gamma \cdot U_{t + 1} \text{更接近真实值} \\
& y_t - R_t = R_t + \gamma \cdot U_{t + 1} - U_t \text{, 称作TD error}
\end{align}
$$
时序差分算法将TD error与步长的乘积作为状态价值的更新量，即：
$$
U_t \gets U_t + \alpha [R_t + \gamma \cdot U_{t + 1} - U_t]
$$

### sarsa算法

可以直接用时序差分算法来估计动作价值函数：
$$
Q(s_t, a_t) \gets Q(s_t, a_t) + \alpha [r_t + \gamma Q(s_{t+1}, a_{t+1}) -Q(s_t, a_t)]
$$
然后用贪婪算法来选取在某个状态下动作价值最大的那个动作，即$arg max_a Q(s, a)$。

这样似乎已经形成了一个完整的强化学习算法：用贪婪算法根据动作价值选取动作来和环境交互，再根据得到的数据用时序差分算法更新动作价值估计。

然而这个简单的算法存在两个需要进一步考虑的问题。

* 如果要用时序差分算法来准确地估计策略的状态价值函数，需要用极大量的样本来进行更新。但实际上可以忽略这一点，直接用一些样本来评估策略，然后就可以更新策略了。可以这么做的原因是策略提升可以在策略评估未完全进行的情况进。
* 如果在策略提升中一直根据贪婪算法得到一个确定性策略，可能会导致某些状态动作对永远没有在序列中出现，以至于无法对其动作价值进行估计，进而无法保证策略提升后的策略比之前的好。简单常用的解决方案是不再一味使用贪婪算法，而是采用一个$\epsilon$-greedy策略：有的$\epsilon$概率采用动作价值最大的那个动作，另外有$1 - \epsilon$的概率从动作空间中随机采取一个动作。

`Sarsa` 的具体算法如下：

> $初始化 Q(s, a)$
>
> $for \ 序列 e = 1 \to e \ do：$
>
> ​        $得到初始状态s$
>
> ​        $用 \epsilon-greedy 策略根据选择当前状态s下的动作a$
>
> ​        $ for\ 时间步t = 1 \to T \  do :$
>
> ​                $ 得到环境反馈的r, s'$
>
> ​                $用\epsilon-greedy 策略根据选择当前状态s'下的动作a'$
>
> ​                $Q(s_t, a_t) \gets Q(s_t, a_t) + \alpha [r_t + \gamma Q(s_{t+1}, a_{t+1}) -Q(s_t, a_t)]$
>
> ​                 $s \gets s', a \gets a'$



### q-learning算法

除了 Sarsa，还有一种非常著名的基于时序差分算法的强化学习算法——Q-learning。Q-learning 和 Sarsa 的最大区别在于 Q-learning 的时序差分更新方式为：

![image-20240909212259058](D:\User\Desktop\current\learn_rl\image\note\image-20240909212259058.png)

Q-learning 算法的具体流程如下：

![image-20240909212340385](D:\User\Desktop\current\learn_rl\image\note\image-20240909212340385.png)

q-learning是在估计$Q^*$，而而 Sarsa 估计当前$\epsilon$-greedy策略的动作价值函数。需要强调的是，Q-learning 的更新并非必须使用当前贪心策略$argmax_aQ(s,a)$采样得到的数据，因为给定任意$(s,a,r,s')$都可以直接根据更新公式来更新$Q$，为了探索，通常使用一个$\epsilon$-greedy策略来与环境交互。Sarsa 必须使用当前$\epsilon$-greedy策略采样得到的数据，因为它的更新中用到的$Q(s',a')$中的$a'$是当前策略在$s'$下的动作。我们称 Sarsa 是**在线策略**（on-policy）算法，称 Q-learning 是**离线策略**（off-policy）算法，这两个概念强化学习中非常重要。

> 称采样数据的策略为**行为策略**（behavior policy），称用这些数据来更新的策略为**目标策略**（target policy）。在线策略（on-policy）算法表示行为策略和目标策略是同一个策略；而离线策略（off-policy）算法表示行为策略和目标策略不是同一个策略。Sarsa 是典型的在线策略算法，而 Q-learning 是典型的离线策略算法。判断二者类别的一个重要手段是看**计算TD target的数据是否来自当前的策略**，如图 所示。具体而言：
>
> - 对于 Sarsa，它的更新公式必须使用来自当前策略采样得到的五元组$(s,a,r,s',a')$，因此它是在线策略学习方法；
> - 对于 Q-learning，它的更新公式使用的是四元组$(s,a,r,a')$来更新当前状态动作对$Q(s,a)$的价值，数据中的$s$和$a$是给定的条件，$r$和$s'$皆由环境采样得到，该四元组并不需要一定是当前策略采样得到的数据，也可以来自行为策略，因此它是离线策略算法。
>
> ![img](D:\User\Desktop\current\learn_rl\image\note\400.78f393db.png)
>
> 离线策略算法能够重复使用过往训练样本，往往具有更小的样本复杂度，也因此更受欢迎。



## policy-base策略学习
