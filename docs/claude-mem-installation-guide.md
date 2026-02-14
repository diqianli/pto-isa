# Claude-Mem 自动安装指南

**版本**: 10.0.6
**更新日期**: 2026-02-14
**适用系统**: Windows, Linux, macOS

---

## 📋 目录

1. [简介](#简介)
2. [系统要求](#系统要求)
3. [自动安装脚本](#自动安装脚本)
4. [手动安装步骤](#手动安装步骤)
5. [验证安装](#验证安装)
6. [使用指南](#使用指南)
7. [故障排除](#故障排除)
8. [高级配置](#高级配置)

---

## 简介

**Claude-Mem** 是 Claude Code 的持久化记忆插件，提供以下功能：

- 🧠 **自动记忆捕获** - 记录所有文件操作、代码编写、命令执行
- 🔍 **语义搜索** - 使用自然语言查询项目历史
- 💾 **跨会话记忆** - 在新会话中自动恢复上下文
- 📊 **Web UI** - 实时查看记忆流和统计信息

**官方网站**: https://docs.claude-mem.ai/introduction
**GitHub 仓库**: https://github.com/thedotmack/claude-mem

---

## 系统要求

### 必需组件

| 组件 | 最低版本 | 安装方法 |
|------|---------|---------|
| **Node.js** | 18.0.0+ | https://nodejs.org |
| **npm** | 自动安装 | 随 Node.js 一起安装 |
| **Claude Code** | 最新版 | 官方 CLI |
| **Git** | 任意版本 | https://git-scm.com |

### 自动安装组件

安装脚本会自动处理以下依赖：

- **Bun** - JavaScript 运行时（自动安装）
- **Python 3.13** - 向量搜索（自动安装）
- **SQLite 3** - 数据存储（内置）

---

## 自动安装脚本

### Windows (PowerShell)

```powershell
# 创建安装脚本
$installScript = @'
# Claude-Mem 自动安装脚本 (Windows)
# 版本: 10.0.6
# 日期: 2026-02-14

Write-Host "=== Claude-Mem 自动安装 ===" -ForegroundColor Cyan

# 1. 检查 Node.js
Write-Host "`n[1/7] 检查 Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version
    Write-Host "✓ Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ 未安装 Node.js，请先安装: https://nodejs.org" -ForegroundColor Red
    exit 1
}

# 2. 安装 Bun
Write-Host "`n[2/7] 安装 Bun 运行时..." -ForegroundColor Yellow
if (-not (Get-Command bun -ErrorAction SilentlyContinue)) {
    curl -fsSL https://bun.sh/install | bash
    $env:BUN_INSTALL = "$HOME/.bun"
    $env:PATH = "$env:BUN_INSTALL/bin:$env:PATH"
    Write-Host "✓ Bun 安装成功" -ForegroundColor Green
} else {
    Write-Host "✓ Bun 已安装" -ForegroundColor Green
}

# 3. 创建插件目录
Write-Host "`n[3/7] 创建插件目录..." -ForegroundColor Yellow
$pluginDir = "$env:USERPROFILE\.claude\plugins\marketplaces\thedotmack"
New-Item -ItemType Directory -Force -Path $pluginDir | Out-Null
Write-Host "✓ 目录创建成功: $pluginDir" -ForegroundColor Green

# 4. 克隆仓库
Write-Host "`n[4/7] 克隆 Claude-Mem 仓库..." -ForegroundColor Yellow
$repoDir = "$pluginDir\claude-mem"
if (Test-Path $repoDir) {
    Write-Host "⚠ 仓库已存在，跳过克隆" -ForegroundColor Yellow
} else {
    git clone https://github.com/thedotmack/claude-mem.git $repoDir
    Write-Host "✓ 仓库克隆成功" -ForegroundColor Green
}

# 5. 安装依赖
Write-Host "`n[5/7] 安装 NPM 依赖..." -ForegroundColor Yellow
Push-Location $repoDir
npm install
Pop-Location
Write-Host "✓ 依赖安装完成" -ForegroundColor Green

# 6. 构建钩子和服务
Write-Host "`n[6/7] 构建钩子和服务..." -ForegroundColor Yellow
Push-Location $repoDir
npm run build
Pop-Location
Write-Host "✓ 构建完成" -ForegroundColor Green

# 7. 启动 Worker 服务
Write-Host "`n[7/7] 启动 Worker 服务..." -ForegroundColor Yellow
$env:BUN_INSTALL = "$HOME/.bun"
$env:PATH = "$env:BUN_INSTALL/bin:$env:PATH"
Push-Location $repoDir
npm run worker:start
Pop-Location
Start-Sleep -Seconds 3
Write-Host "✓ Worker 服务启动" -ForegroundColor Green

# 8. 配置 MCP 服务器
Write-Host "`n[8/8] 配置 MCP 服务器..." -ForegroundColor Yellow
$configDir = "$env:APPDATA\Claude"
New-Item -ItemType Directory -Force -Path $configDir | Out-Null

$configPath = "$configDir\claude_desktop_config.json"
$configContent = @"
{
  "mcpServers": {
    "claude-mem": {
      "command": "node",
      "args": ["$repoDir\plugin\scripts\mcp-server.cjs"],
      "env": {
        "CLAUDE_MEM_HOME": "$env:USERPROFILE\.claude-mem"
      }
    }
  }
}
"@

Set-Content -Path $configPath -Value $configContent
Write-Host "✓ MCP 配置文件创建: $configPath" -ForegroundColor Green

# 完成
Write-Host "`n=== 安装完成！ ===" -ForegroundColor Green
Write-Host "Worker 服务运行在: http://localhost:37777" -ForegroundColor Cyan
Write-Host "请重启 Claude Code 以启用 MCP 工具" -ForegroundColor Yellow
'@

$installScript | Out-File -FilePath "install-claude-mem.ps1" -Encoding UTF8
Write-Host "安装脚本已创建: install-claude-mem.ps1" -ForegroundColor Green
#@

# 执行安装脚本
./install-claude-mem.ps1
```

### Linux / macOS (Bash)

```bash
#!/bin/bash
# Claude-Mem 自动安装脚本 (Linux/macOS)
# 版本: 10.0.6
# 日期: 2026-02-14

set -e

echo "=== Claude-Mem 自动安装 ==="

# 1. 检查 Node.js
echo ""
echo "[1/7] 检查 Node.js..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "✓ Node.js: $NODE_VERSION"
else
    echo "✗ 未安装 Node.js，请先安装: https://nodejs.org"
    exit 1
fi

# 2. 安装 Bun
echo ""
echo "[2/7] 安装 Bun 运行时..."
if ! command -v bun &> /dev/null; then
    curl -fsSL https://bun.sh/install | bash
    export BUN_INSTALL="$HOME/.bun"
    export PATH="$BUN_INSTALL/bin:$PATH"
    echo "✓ Bun 安装成功"
else
    echo "✓ Bun 已安装"
fi

# 3. 创建插件目录
echo ""
echo "[3/7] 创建插件目录..."
PLUGIN_DIR="$HOME/.claude/plugins/marketplaces/thedotmack"
mkdir -p "$PLUGIN_DIR"
echo "✓ 目录创建成功: $PLUGIN_DIR"

# 4. 克隆仓库
echo ""
echo "[4/7] 克隆 Claude-Mem 仓库..."
REPO_DIR="$PLUGIN_DIR/claude-mem"
if [ -d "$REPO_DIR" ]; then
    echo "⚠ 仓库已存在，跳过克隆"
else
    git clone https://github.com/thedotmack/claude-mem.git "$REPO_DIR"
    echo "✓ 仓库克隆成功"
fi

# 5. 安装依赖
echo ""
echo "[5/7] 安装 NPM 依赖..."
cd "$REPO_DIR"
npm install
echo "✓ 依赖安装完成"

# 6. 构建钩子和服务
echo ""
echo "[6/7] 构建钩子和服务..."
npm run build
echo "✓ 构建完成"

# 7. 启动 Worker 服务
echo ""
echo "[7/7] 启动 Worker 服务..."
export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"
npm run worker:start
sleep 3
echo "✓ Worker 服务启动"

# 8. 配置 MCP 服务器
echo ""
echo "[8/8] 配置 MCP 服务器..."
CONFIG_DIR="$HOME/.config/Claude"
mkdir -p "$CONFIG_DIR"

CONFIG_PATH="$CONFIG_DIR/claude_desktop_config.json"
cat > "$CONFIG_PATH" << EOF
{
  "mcpServers": {
    "claude-mem": {
      "command": "node",
      "args": ["$REPO_DIR/plugin/scripts/mcp-server.cjs"],
      "env": {
        "CLAUDE_MEM_HOME": "$HOME/.claude-mem"
      }
    }
  }
}
EOF

echo "✓ MCP 配置文件创建: $CONFIG_PATH"

# 完成
echo ""
echo "=== 安装完成！ ==="
echo "Worker 服务运行在: http://localhost:37777"
echo "请重启 Claude Code 以启用 MCP 工具"
```

保存为 `install-claude-mem.sh` 并执行：
```bash
chmod +x install-claude-mem.sh
./install-claude-mem.sh
```

---

## 手动安装步骤

如果自动脚本失败，请按以下步骤手动安装：

### 步骤 1: 检查环境

```bash
# 检查 Node.js 版本（需要 >= 18.0.0）
node --version

# 检查 npm 版本
npm --version
```

### 步骤 2: 安装 Bun

```bash
# 使用官方安装脚本
curl -fsSL https://bun.sh/install | bash

# 添加到 PATH（添加到 ~/.bashrc 或 ~/.zshrc）
export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"

# 验证安装
bun --version
```

### 步骤 3: 克隆仓库

```bash
# 创建插件目录
mkdir -p ~/.claude/plugins/marketplaces/thedotmack

# 克隆仓库
git clone https://github.com/thedotmack/claude-mem.git \
  ~/.claude/plugins/marketplaces/thedotmack/claude-mem
```

### 步骤 4: 安装依赖

```bash
cd ~/.claude/plugins/marketplaces/thedotmack/claude-mem
npm install
```

预期输出：
```
added 517 packages, and audited 518 packages
```

### 步骤 5: 构建钩子和服务

```bash
npm run build
```

预期输出：
```
✓ React viewer built successfully
✓ worker-service built
✓ mcp-server built
✓ context-generator built
```

### 步骤 6: 启动 Worker 服务

```bash
npm run worker:start
```

预期输出：
```
[SETTINGS] Created settings file with defaults
{"continue":true,"suppressOutput":true,"status":"ready"}
```

### 步骤 7: 配置 MCP 服务器

#### Windows

创建 `%APPDATA%\Claude\claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "claude-mem": {
      "command": "node",
      "args": ["C:/Users/YourUsername/.claude/plugins/marketplaces/thedotmack/claude-mem/plugin/scripts/mcp-server.cjs"],
      "env": {
        "CLAUDE_MEM_HOME": "C:/Users/YourUsername/.claude-mem"
      }
    }
  }
}
```

#### Linux / macOS

创建 `~/.config/Claude/claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "claude-mem": {
      "command": "node",
      "args": ["/home/yourusername/.claude/plugins/marketplaces/thedotmack/claude-mem/plugin/scripts/mcp-server.cjs"],
      "env": {
        "CLAUDE_MEM_HOME": "/home/yourusername/.claude-mem"
      }
    }
  }
}
```

**重要**：替换 `YourUsername` 为实际用户名！

---

## 验证安装

### 1. 检查 Worker 服务状态

```bash
cd ~/.claude/plugins/marketplaces/thedotmack/claude-mem
npm run worker:status
```

预期输出：
```
Worker is running
  PID: <进程ID>
  Port: 37777
  Started: <时间戳>
```

### 2. 访问 Web UI

在浏览器打开：**http://localhost:37777**

您应该看到 Claude-Mem 的 Web UI 界面。

### 3. 检查 API 健康状态

```bash
curl http://localhost:37777/api/health
```

预期输出：
```json
{
  "status": "ok",
  "version": "10.0.6",
  "mcpReady": true
}
```

### 4. 检查数据库

```bash
ls -lah ~/.claude-mem/
```

应该看到：
```
claude-mem.db
settings.json
logs/
```

### 5. 重启 Claude Code 并验证 MCP 工具

关闭并重新启动 Claude Code，然后运行：

```bash
claude --dangerously-skip-permissions "列出所有可用的 MCP 工具"
```

应该看到类似输出：

```
### N. Claude-Mem Memory System
- `mcp__claude-mem__search` - Search memory index
- `mcp__claude-mem__timeline` - Get chronological context
- `mcp__claude-mem__get_observations` - Fetch full details
- `mcp__claude-mem__save_memory` - Manually save memory
```

---

## 使用指南

### 自动记忆捕获

重启 Claude Code 后，所有操作将自动记录：

- ✅ **文件读取** - 自动记录文件内容和上下文
- ✅ **代码编写** - 记录所有生成的代码
- ✅ **命令执行** - 记录 Bash 命令和输出
- ✅ **Bug 修复** - 自动标记问题和解决方案
- ✅ **决策记录** - 捕获技术决策和理由

### MCP 工具使用

#### 1. 搜索记忆

```javascript
// 基础搜索
search(query="PTO-ISA 优化", limit=10)

// 按类型过滤
search(query="bugfix", type="bugfix", limit=20)

// 按日期过滤
search(query="Flash Attention", date="2026-01-20", limit=10)
```

#### 2. 查看时序上下文

```javascript
// 查看特定观察的前后活动
timeline(observation_id=123)

// 查看查询的时序上下文
timeline(query="循环融合优化")
```

#### 3. 获取完整详情

```javascript
// 批量获取（推荐）
get_observations(ids=[123, 456, 789])

// 单个获取
get_observations(ids=[123])
```

#### 4. 手动保存记忆

```javascript
// 保存重要决策
save_memory(
  text="PTO-ISA 使用两阶段依赖解析：Memory 域用原子，Register 域用标量",
  title="两阶段依赖解析设计",
  type="decision",
  project="pto-isa"
)

// 保存里程碑
save_memory(
  text="PTO-ISA 项目记忆文件创建完成，14 KB",
  title="项目文档完成",
  type="milestone"
)
```

### Web UI 功能

访问 **http://localhost:37777** 可以：

- 📊 **实时记忆流** - 查看所有捕获的活动
- 🔍 **搜索界面** - 自然语言搜索历史
- 📈 **统计信息** - 会话统计、记忆数量
- ⚙️ **设置面板** - 调整配置和行为
- 📄 **API 文档** - 查看 REST API 端点

### Worker 服务管理

```bash
cd ~/.claude/plugins/marketplaces/thedotmack/claude-mem

# 查看状态
npm run worker:status

# 停止服务
npm run worker:stop

# 重启服务
npm run worker:restart

# 查看日志
npm run worker:logs

# 实时跟踪日志
npm run worker:tail
```

---

## 故障排除

### 问题 1: Worker 服务未运行

**症状**：
```
Worker is not running
```

**解决方案**：
```bash
cd ~/.claude/plugins/marketplaces/thedotmack/claude-mem

# 查看错误日志
cat ~/.claude-mem/logs/claude-mem-*.log | tail -50

# 重启服务
npm run worker:restart

# 如果仍然失败，手动启动
npm run worker:start
```

### 问题 2: MCP 工具未加载

**症状**：重启后看不到 `claude-mem` MCP 工具

**解决方案**：

1. **验证配置文件路径**：
   ```bash
   # Windows
   cat "%APPDATA%\Claude\claude_desktop_config.json"

   # Linux/macOS
   cat ~/.config/Claude/claude_desktop_config.json
   ```

2. **验证 MCP 服务器文件存在**：
   ```bash
   ls -la ~/.claude/plugins/marketplaces/thedotmack/claude-mem/plugin/scripts/mcp-server.cjs
   ```

3. **手动测试 MCP 服务器**：
   ```bash
   node ~/.claude/plugins/marketplaces/thedotmack/claude-mem/plugin/scripts/mcp-server.cjs
   ```

4. **检查路径是否正确**（替换用户名）：
   ```json
   {
     "mcpServers": {
       "claude-mem": {
         "command": "node",
         "args": ["完整路径到 mcp-server.cjs"],
         "env": {
           "CLAUDE_MEM_HOME": "完整路径到 .claude-mem"
         }
       }
     }
   }
   ```

### 问题 3: Bun 命令未找到

**症状**：
```
'bun' 不是内部或外部命令
```

**解决方案**：

```bash
# 重新安装 Bun
curl -fsSL https://bun.sh/install | bash

# 添加到 PATH
echo 'export BUN_INSTALL="$HOME/.bun"' >> ~/.bashrc
echo 'export PATH="$BUN_INSTALL/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 验证
bun --version
```

Windows 下手动安装：
1. 访问 https://bun.sh
2. 下载 Windows 安装程序
3. 运行安装程序
4. 重启终端

### 问题 4: 端口 37777 被占用

**症状**：
```
Error: Port 37777 already in use
```

**解决方案**：

```bash
# 查找占用进程
# Windows
netstat -ano | findstr :37777

# Linux/macOS
lsof -i :37777

# 终止进程或修改端口
# 编辑 ~/.claude-mem/settings.json
# 修改 CLAUDE_MEM_WORKER_PORT
```

### 问题 5: 数据库锁定

**症状**：
```
database is locked
```

**解决方案**：

```bash
# 停止 Worker
npm run worker:stop

# 删除锁文件
rm ~/.claude-mem/claude-mem.db-shm
rm ~/.claude-mem/claude-mem.db-wal

# 重启 Worker
npm run worker:start
```

### 问题 6: 内存不足

**症状**：
```
JavaScript heap out of memory
```

**解决方案**：

编辑 Worker 启动脚本，增加堆内存：
```bash
# 在启动命令前添加
export NODE_OPTIONS="--max-old-space-size=4096"
```

---

## 高级配置

### 自定义设置

编辑 `~/.claude-mem/settings.json`：

```json
{
  // AI 模型选择
  "CLAUDE_MEM_MODEL": "claude-sonnet-4-5",

  // 上下文观察数量
  "CLAUDE_MEM_CONTEXT_OBSERVATIONS": "50",

  // Worker 端口
  "CLAUDE_MEM_WORKER_PORT": "37777",

  // Worker 主机
  "CLAUDE_MEM_WORKER_HOST": "127.0.0.1",

  // 跳过的工具（不记录）
  "CLAUDE_MEM_SKIP_TOOLS": "ListMcpResourcesTool,SlashCommand,Skill,TodoWrite,AskUserQuestion",

  // 日志级别
  "CLAUDE_MEM_LOG_LEVEL": "INFO",

  // 数据目录
  "CLAUDE_MEM_DATA_DIR": "C:\\Users\\YourUsername\\.claude-mem",

  // 排除的项目
  "CLAUDE_MEM_EXCLUDED_PROJECTS": "secret-project,private-repo",

  // 记忆类型
  "CLAUDE_MEM_CONTEXT_OBSERVATION_TYPES": "bugfix,feature,refactor,discovery,decision,change"
}
```

### 排除敏感信息

使用 `<private>` 标签：

```python
# <private> 不要记录这个 API 密钥
api_key = "sk-xxxxx"

# <private> 敏感配置文件
config = load_config("secrets.conf")
```

### 自定义钩子

编辑 `~/.claude/plugins/marketplaces/thedotmack/claude-mem/plugin/hooks/hooks.json` 添加自定义逻辑。

详见：[Hooks Architecture](https://docs.claude-mem.ai/hooks)

---

## 卸载

### Windows

```powershell
# 停止 Worker
cd ~/plugins/marketplaces/thedotmack/claude-mem
npm run worker:stop

# 删除插件目录
Remove-Item -Recurse -Force ~/.claude/plugins/marketplaces/thedotmack/claude-mem

# 删除配置文件
Remove-Item -Force $env:APPDATA\Claude\claude_desktop_config.json

# 删除数据目录（可选）
Remove-Item -Recurse -Force ~/.claude-mem
```

### Linux / macOS

```bash
# 停止 Worker
cd ~/.claude/plugins/marketplaces/thedotmack/claude-mem
npm run worker:stop

# 删除插件目录
rm -rf ~/.claude/plugins/marketplaces/thedotmack/claude-mem

# 删除配置文件
rm ~/.config/Claude/claude_desktop_config.json

# 删除数据目录（可选）
rm -rf ~/.claude-mem
```

---

## 更新

### 更新到最新版本

```bash
cd ~/.claude/plugins/marketplaces/thedotmack/claude-mem

# 拉取最新代码
git pull origin main

# 安装新依赖
npm install

# 重新构建
npm run build

# 重启 Worker
npm run worker:restart
```

### 检查当前版本

```bash
npm run worker:status
# 输出中的 version 字段
```

或访问 http://localhost:37777/api/health

---

## 性能优化

### 减少内存占用

```json
{
  "CLAUDE_MEM_CONTEXT_OBSERVATIONS": "30",
  "CLAUDE_MEM_CONTEXT_SESSION_COUNT": "5"
}
```

### 加快搜索速度

```json
{
  "CLAUDE_MEM_CONTEXT_FULL_COUNT": "3",
  "CLAUDE_MEM_CONTEXT_FULL_FIELD": "summary"
}
```

### 禁用自动保存

```json
{
  "CLAUDE_MEM_FOLDER_CLAUDEMD_ENABLED": "false"
}
```

---

## 安全注意事项

1. **敏感数据** - 使用 `<private>` 标签保护密码、API 密钥
2. **本地存储** - 所有数据存储在本地 SQLite 数据库
3. **网络访问** - Worker 默认仅绑定 127.0.0.1
4. **备份** - 定期备份 `~/.claude-mem/` 目录
5. **权限** - 配置文件权限限制为仅用户可读写

---

## 参考资料

- **官方文档**: https://docs.claude-mem.ai
- **GitHub**: https://github.com/thedotmack/claude-mem
- **问题报告**: https://github.com/thedotmack/claude-mem/issues
- **社区讨论**: https://reddit.com/r/ClaudeCode

---

## 许可证

Claude-Mem 使用 **GNU Affero General Public License v3.0 (AGPL-3.0)**

详见：[LICENSE](https://github.com/thedotmack/claude-mem/blob/main/LICENSE)

---

**文档版本**: 1.0
**最后更新**: 2026-02-14
**作者**: 基于 Claude-Mem v10.0.6 官方文档整理
