# Vercel Labs Skills 安装指南（离线/代理方案）

## 问题诊断

您遇到的错误是因为：
1. 网络无法直接访问 GitHub (防火墙/代理问题)
2. npm 无法使用 git 克隆仓库

## 解决方案

### 方案1: 手动下载安装（推荐）

#### 步骤1: 手动下载 skills 包

**方法A - 使用浏览器下载**
1. 访问：https://github.com/vercel-labs/skills
2. 点击绿色 "Code" 按钮
3. 选择 "Download ZIP"
4. 保存到本地目录（如 `Downloads/skills.zip`）

**方法B - 使用镜像站点**
```bash
# 如果有代理或镜像访问
wget https://mirror.ghproxy.com/https://github.com/vercel-labs/skills/archive/refs/heads/main.zip -O skills.zip
```

#### 步骤2: 解压并安装

```bash
# 解压文件
cd /tmp
unzip ~/Downloads/skills.zip
cd skills-main

# 查看内容
ls -la

# 安装依赖
npm install

# 构建项目
npm run build

# 全局安装
npm install -g .
```

#### 步骤3: 验证安装

```bash
# 检查是否安装成功
which skills
skills --help
```

---

### 方案2: 配置代理后安装

如果您有代理服务器：

#### 配置 git 代理
```bash
# 设置 HTTP 代理
git config --global http.proxy http://proxy-server:port
git config --global https.proxy http://proxy-server:port

# 或者设置 SOCKS 代理
git config --global core.gitproxy 'socks5://127.0.0.1:1080'
```

#### 配置 npm 代理
```bash
npm config set proxy http://proxy-server:port
npm config set https-proxy http://proxy-server:port
```

#### 重新尝试安装
```bash
# 安装 skills
npm install -g https://github.com/vercel-labs/skills.git --yes
```

#### 取消代理（如果需要）
```bash
# 取消 git 代理
git config --global --unset http.proxy
git config --global --unset https.proxy

# 取消 npm 代理
npm config delete proxy
npm config delete https-proxy
```

---

### 方案3: 使用 GitHub 镜像

#### 步骤1: 克隆镜像仓库

```bash
# 使用 ghproxy 镜像
git clone https://mirror.ghproxy.com/https://github.com/vercel-labs/skills.git

# 或使用 gitclone 镜像
git clone https://gitclone.github.com/github/vercel-labs/skills

# 或使用其他镜像
git clone https://github.com.cnpmjs.org/vercel-labs/skills.git
```

#### 步骤2: 进入目录并安装
```bash
cd skills
npm install
npm run build
npm install -g .
```

---

### 方案4: 使用已创建的本地 Skills 配置（最简单）

**注意**: 我已经为您在 PTO-ISA 项目中创建了自定义 skills 配置，无需安装 Vercel Labs skills！

#### 使用现有的自定义 Skills

您的项目已经有4个自定义skills：

1. **`/pto-analyze`** - 分析PTO-ISA架构
2. **`/pto-optimize`** - 优化调度器性能
3. **`/pto-trace`** - 追踪执行流程
4. **`/pto-benchmark`** - 运行基准测试

#### 配置文件位置
```
E:\cccode\pto-isa\.claude\config.json
E:\cccode\pto-isa\.claude\README.md
```

#### 直接使用示例
```
# 使用方式1: 直接对话
"使用pto-analyze skill分析调度算法"

# 使用方式2: 自然语言
"帮我优化pto_scheduler.c的性能"

# 使用方式3: 明确指定
"运行pto-benchmark进行性能测试"
```

---

## Vercel Labs Skills 包含的功能

如果您仍然想安装 Vercel Labs skills，它包含以下功能：

### 主要 Skills
- **Code Analysis** - 代码分析
- **Bug Detection** - Bug检测
- **Performance Optimization** - 性能优化
- **Code Review** - 代码审查
- **Documentation** - 文档生成

### 这些功能与PTO-ISA项目的关系

**PTO-ISA项目的自定义skills**更加专业和定制化：
- ✅ 专门针对PTO-ISA架构
- ✅ 深入理解调度算法
- ✅ 包含性能分析工具
- ✅ 集成BGEMM示例

**而Vercel Labs skills**是通用的：
- 🔧 通用代码分析
- 🔧 基础性能优化
- 🔧 标准代码审查

**建议**: 对于PTO-ISA项目，使用已创建的自定义skills效果更好！

---

## 故障排除

### 问题1: npm install 失败

**错误**: `npm ERR! command failed`

**解决**:
```bash
# 清除 npm 缓存
npm cache clean --force

# 使用淘宝镜像
npm config set registry https://registry.npmmirror.com

# 重新安装
npm install
```

### 问题2: git clone 失败

**错误**: `Failed to connect to github.com`

**解决**:
```bash
# 检查网络连接
ping github.com

# 尝试使用 IP 地址
# 在 C:\Windows\System32\drivers\etc\hosts 添加：
# 140.82.112.4 github.com
```

### 问题3: 权限错误

**错误**: `Permission denied (publickey)`

**解决**:
```bash
# 使用 HTTPS 而不是 SSH
git clone https://github.com/vercel-labs/skills.git

# 而不是
git clone git@github.com:vercel-labs/skills.git
```

---

## 推荐方案

### 🎯 最佳方案：使用已创建的自定义 Skills

**原因**:
1. ✅ 无需安装任何东西
2. ✅ 专门为PTO-ISA项目定制
3. ✅ 已经配置完成，立即可用
4. ✅ 包含详细的使用文档

**立即开始使用**:
```
你: "使用pto-analyze分析pto_scheduler.c"
我: [开始分析调度器实现...]
```

---

## 联系支持

如果以上方案都无法解决问题：

1. 检查防火墙设置
2. 联系网络管理员
3. 尝试使用手机热点
4. 或者直接使用项目已配置的自定义skills

---

**文档版本**: 1.0
**最后更新**: 2025-02-10
**状态**: ✅ 自定义skills已就绪，无需安装外部包
