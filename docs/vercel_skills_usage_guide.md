# Vercel Labs Skills 使用指南

## ✅ 安装状态

**Skill**: `find-skills`
**状态**: ✅ 已成功安装
**位置**: `~/.agents/skills/find-skills`
**目标环境**: Claude Code, Cursor
**安装模式**: 全局安装

---

## 🎯 find-skills Skill 功能

`find-skills` 是一个强大的文件搜索skill，可以：

- 🔍 **搜索代码仓库**：在GitHub上搜索相关文件
- 📝 **查找实现示例**：发现最佳实践和代码模式
- 🎓 **学习资源**：找到教程和文档
- ⚡ **快速定位**：无需离开编辑器即可搜索

---

## 📖 使用方法

### 方法1: 通过对话使用（推荐）

```
你: 使用 find-skills 搜索 "task scheduler" 的实现
我: [调用 find-skills skill]
    [返回搜索结果和相关资源]
```

### 方法2: 直接命令行使用

```bash
# 在项目目录中运行
cd /e/cccode/pto-isa

# 搜索调度相关文件
npx skills find "task scheduler"

# 搜索算法相关内容
npx skills find "dependency resolution"

# 搜索优化相关内容
npx skills find "performance optimization"
```

---

## 💡 使用示例

### 示例1: 搜索调度算法实现

**对话方式**:
```
你: 使用 find-skills 搜索 "C++ task scheduler implementation"
我: [调用 find-skills 并返回结果]
```

**命令行方式**:
```bash
npx skills find "C++ task scheduler"
```

**预期结果**:
- 找到GitHub上的调度器实现项目
- 发现参考代码和最佳实践
- 提供学习资源和示例

---

### 示例2: 搜索性能优化技巧

```
你: 用 find-skills 搜索 "lock-free queue" 性能优化
我: [搜索并返回优化建议和代码示例]
```

---

## 🔧 在 PTO-ISA 项目中的实际应用

### 应用场景1: 研究优化方案

```bash
cd /e/cccode/pto-isa

# 搜索无锁队列实现
npx skills find "lock-free queue c++"

# 搜索工作窃取调度
npx skills find "work stealing scheduler"

# 搜索内存管理优化
npx skills find "memory optimization ring buffer"
```

### 应用场景2: 查找参考实现

```bash
# 搜索高性能调度器
npx skills find "high performance scheduler"

# 搜索并发控制
npx skills find "concurrency control atomic"

# 搜索依赖解析
npx skills find "dependency resolution algorithm"
```

---

## 📊 与现有 Skills 的对比

| 功能 | Vercel find-skills | PTO-ISA 自定义 Skills |
|------|-------------------|-------------------|
| **范围** | 全GitHub代码库 | PTO-ISA项目专用 |
| **用途** | 发现新代码库 | 分析现有代码 |
| **优势** | 全球代码搜索 | 深度项目知识 |
| **使用场景** | 研究技术方案 | 日常开发工作 |

**最佳实践**: 结合使用两者！

```
# 研究阶段 - 使用 find-skills
npx skills find "optimization techniques"

# 开发阶段 - 使用 PTO-ISA skills
使用 pto-analyze skill 分析调度器

# 学习阶段 - 使用 find-skills
npx skills find "scheduler tutorial"
```

---

## 🎓 高级用法

### 搜索特定编程语言

```bash
# C++ 调度器
npx skills find "c++ scheduler language:c++"

# Python 优化
npx skills find "performance optimization language:python"

# Rust 并发
npx skills find "concurrent rust language:rust"
```

### 搜索特定仓库

```bash
# 搜索特定组织
npx skills find "scheduler org:torvalds"

# 搜索特定语言
npx skills find "scheduler language:c++ stars:>100"
```

---

## 🔍 常用搜索查询

### PTO-ISA 相关查询

```bash
# 1. 调度算法
npx skills find "task scheduling algorithm"
npx skills find "dependency resolution"
npx skills find "load balancing"
npx skills find "fair scheduling"

# 2. 性能优化
npx skills find "reduce scheduling overhead"
npx skills find "optimize lock contention"
npx skills find "cache-friendly data structure"

# 3. 并发编程
npx skills find "atomic operations"
npx skills find "memory barriers"
npx skills find "lock-free programming"
npx skills find "wait-free queue"
```

### 通用技术查询

```bash
# 算法和数据结构
npx skills find "b-tree implementation"
npx skills find "hash table optimization"
npx skills find "graph traversal"

# 系统编程
npx skills find "system programming patterns"
npx skills find "embedded systems optimization"

# 架构设计
npx skills find "microservices patterns"
npx skills find "event-driven architecture"
```

---

## ⚡ 性能优化搜索实例

### 问题: 调度器性能瓶颈

**对话方式**:
```
你: 使用 find-skills 搜索如何减少调度开销
我: [调用 find-skills]

搜索结果:
1. "Batch scheduling reduce overhead"
2. "Lock-free queue implementation"
3. "Wakeup optimization scheduler"

推荐方案: 批量处理依赖解析
```

**命令行方式**:
```bash
npx skills find "reduce scheduler overhead"
```

---

## 🌟 与 PTO-ISA 集成工作流

### 完整研究流程

```
1. 问题识别
   → 使用 pto-analyze skill 分析当前架构

2. 方案研究
   → 使用 find-skills 搜索全球最佳实践
   npx skills find "scheduler optimization"

3. 方案设计
   → 结合两种skill的研究结果
   → 设计定制化解决方案

4. 方案实现
   → 使用 pto-optimize skill 指导实现

5. 验证测试
   → 使用 pto-benchmark skill 性能测试
```

---

## 💻 实际示例

### 示例: 优化就绪队列

```bash
# Step 1: 使用 find-skills 研究
cd /e/cccode/pto-isa
npx skills find "ready queue optimization"

# Step 2: 查看PTO-ISA当前实现
cat ref_runtime/src/runtime/rt2/runtime/pto_scheduler.c

# Step 3: 结合研究结果提出优化
使用 pto-optimize skill 生成优化建议
```

---

## 📈 搜索技巧

### 技巧1: 使用更具体的关键词

❌ 太宽泛:
```bash
npx skills find "optimization"
```

✅ 更具体:
```bash
npx skills find "reduce atomic operations"
npx skills find "minimize cache misses"
```

### 技巧2: 组合关键词

```bash
# 同时搜索多个概念
npx skills find "scheduler performance c++"
```

### 技巧3: 搜索实现细节

```bash
# 搜索具体的实现模式
npx skills find "circular buffer implementation lock-free"
npx skills find "dependency graph traversal"
```

---

## 🎯 快速参考

### 常用命令

```bash
# 列出已安装的skills
npx skills list -g

# 搜索代码
npx skills find "关键词"

# 交互式搜索
npx skills find

# 更新skills
npx skills update

# 查看帮助
npx skills --help
```

### PTO-ISA 项目常用搜索

```bash
cd /e/cccode/pto-isa

# 调度相关
npx skills find "task scheduling"
npx skills find "dependency resolution"
npx skills find "load balancing"

# 性能优化
npx skills find "performance optimization"
npx skills find "reduce overhead"
npx skills find "lock-free programming"

# 并发编程
npx skills find "atomic operations"
npx skills find "memory ordering"
npx skills find "concurrency patterns"
```

---

## 🆚 故障排除

### 问题1: 搜索无结果

**可能原因**:
- 关键词太具体
- 代码库未索引
- 网络连接问题

**解决方法**:
```bash
# 使用更通用的关键词
npx skills find "optimization"  # 而不是 "reduce overhead"

# 检查网络连接
ping github.com

# 稍后重试
npx skills check
npx skills update
```

### 问题2: 命令不工作

**检查安装**:
```bash
# 验证skill已安装
npx skills list -g | grep find-skills

# 应该看到:
# find-skills  ~/.agents/skills/find-skills
```

**手动调用**:
```bash
# 直接调用skill
~/.agents/skills/find-skills/SKILL.md "关键词"
```

---

## 📚 相关资源

### Vercel Labs Skills 仓库
- **GitHub**: https://github.com/vercel-labs/skills
- **文档**: https://github.com/vercel-labs/skills/tree/main/README.md

### PTO-ISA Skills 配置
- **配置文件**: `.claude/config.json`
- **使用文档**: `.claude/README.md`
- **自定义skills**:
  - pto-analyze
  - pto-optimize
  - pto-trace
  - pto-benchmark

---

## 🎉 总结

### ✅ 已安装
- **find-skills**: ✅ 全局安装成功
- **位置**: `~/.agents/skills/find-skills`
- **环境**: Claude Code, Cursor

### 🎯 主要用途
1. **技术研究**: 搜索GitHub上的最佳实践
2. **代码学习**: 发现优秀的实现案例
3. **问题解决**: 找到类似问题的解决方案
4. **持续学习**: 保持技术前沿

### 💡 推荐工作流
```
PTO-ISA 自定义 Skills  +  Vercel find-skills
    ↓                        ↓
深度项目分析        +  全球最佳实践
    ↓                        ↓
    定制化解决方案
```

---

**文档版本**: 1.0
**更新日期**: 2025-02-10
**状态**: ✅ find-skills skill 已安装并就绪
