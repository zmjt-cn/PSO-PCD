本软件享有软件著作权：基于粒子群算法的多晶超硬材料超精密抛光路径规划软件, 北京：2024SR0182300, 中国
This software is subject to copyright: Ultra-precision polishing path planning software for polycrystalline superhard materials based on the particle swarm organization, Beijing: 2024SR0182300, China.

zh-cn
{
    作者：任亚斌(zmjt-cn)，杨波
    地址：河北工业大学机械工程学院，天津 300401，中国
    邮箱：<s1345358@126.com>，<boyang@hebut.edu.cn>
    网址：https://github.com/zmjt-cn/PSO-PCD

    用户协议：
    {
        在开始使用本软件(代码)之前，您需要了解并认可一些内容。继续使用本软件(代码)意味着您已认可以下内容：
        1.本软件(代码)遵循Apache License 2.0协议。程序中PSO部分代码来自TimePickerWang撰写的MachineLearning库，
            PSO代码源：https://github.com/TimePickerWang/MachineLearning/blob/master/MachineLearning/OptAlgorithm/PSO.py
        2.本软件(代码)为免费软件(代码)，任何组织或个人不得将软件(代码)的任何内容用于商业行为。
        3.作者享有本软件(代码)著作权，使用请注明来源。
        4.用户协议以最新发布版为准。
    }

    ================================
    ===========PSO_PCD.py===========
    ================================
    =========Version V1.1.2=========
    ============20240123============

    PSO_PCD简介：
    {
        基于粒子群算法的多晶超硬材料超精密抛光路径规划软件(PSO_PCD)。
        
        >>程序开发背景：
            金刚石不同晶向的力学性质存在差异，抛光时展现出明显的各向异性，会对抛光面粗糙度(Sa)产生显著影响。
            为深入研究多晶材料的表面抛光机理，开发了本程序。以超硬材料典型代表，金刚石为例，基于多晶金刚石
            (PCD)的抛光面晶粒信息，结合金刚石各向异性抛光数据库，多次变换抛光方向降低其表面Ra值，实现PCD超
            精密抛光。研究者可使用该软件指导超硬多晶材料的超精密抛光，获得超光滑表面。

        >>程序介绍：
            基于粒子群算法的多晶超硬材料超精密抛光路径规划软件可生成不同表面粗糙度PCD模型并规划抛光路径，
            给出各路径抛光时间。根据单晶和多晶的划痕实验，建立了包含各类晶面晶向的材料去除率数据库。使用
            随机生成法生成具有一定数量晶粒，不同晶面分布与晶向排列构成的显著高度差异的NPD模型，该初始模型
            的表面粗糙度(Sa)设定为大于30 nm。结合PCD模型晶面分布情况与抛光数据库，构造表面粗糙度与各抛光
            方向对应的抛光时间关系。取15°做为变化抛光方向间隔，次数为13次，根据晶体结构的对称性，可覆盖到
            所有晶粒的易抛光方向。随后构建分段粒子群算法(PSO)计算粗糙度最小时的抛光时间。不断优化粒子最优
            位置与全局最优位置，最终当表面粗糙度值小于0.5 nm时，输出最优抛光路径和对应抛光时间。

        >>程序已经实现的功能：
            1.基于粒子群算法构建了多晶超硬材料超精密抛光路径规划软件。
            2.验证了纳米多晶金刚石转向抛光理论的可行性，初步实现了小型纳米多晶金刚石的转向抛光。
            3.更多详细信息见程序使用说明。

        >>程序运行必要的文件：
            “PSO_PCD.py”脚本文件和“Rate_data”数据库文件。

        >>程序使用说明：
            本软件是使用粒子群算法作为核心构建的，运行软件需要在本地安装金刚石抛光数据库，以及安装必要的
            numpy环境包。程序会构建由金刚石(100)、(110)、(111)面组成的共30个晶粒的PCD模型，模型中的晶
            面构成比例、晶向排列情况、晶粒尺寸以及晶粒高度均为随机生成，默认生成模型次数为20次，模型表面
            粗糙度初始为大于30 nm，计算过程中会同步输出当前PSO算法使用次数、最佳适应值、表面粗糙度值以及
            各晶向抛光时间，当表面粗糙度Ra值小于0.5 nm时，模型完成抛光路径规划，进行下一个NPD模型的构建
            与路径规划计算。完成20次计算后程序最终将生成“DataOutput.txt”文件，该文件保存了20个NPD模型的
            抛光路径规划与抛光时间，抛光前后模型表面粗糙度、晶粒高度，以及晶粒尺寸。

        >>程序默认使用参数（可根据实际修改）介绍：
            转向间隔为15°，抛光方向总数为13，生成模型次数为20次，晶粒总数为30，平均晶粒尺寸100 nm，
            晶粒尺寸正态分布标准差为100 nm2，平均晶粒高度5000 nm，晶粒高度正态分布标准差为50 nm，深度
            间隔ΔD为0.5 nm；PSO使用粒子维度为13维，粒子总数为20，迭代次数为401次，粒子位置限制大于0，
            粒子最大速度为1，截止条件1×10-4，路径规划终止条件为当前Sa小于0.5 nm。
    }
    感谢您的使用，欢迎您的指正!
}
