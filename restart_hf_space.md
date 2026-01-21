# 🔄 重启 HuggingFace Space

## 问题
代码已同步到 HuggingFace，但应用仍运行旧版本，logs 未更新。

## 解决方案：重启 Space

### 方法 1：通过 Web UI 重启（推荐）

1. **访问 Space 设置页面**:
   https://huggingface.co/spaces/AnthonyInBC/Anthony_space/settings

2. **找到 "Restart" 按钮**:
   - 在页面顶部或底部
   - 点击 **"Restart"** 或 **"Restart Space"**

3. **等待重启完成**:
   - 通常需要 1-2 分钟
   - 观察 "Logs" 标签页看重启过程

### 方法 2：通过 Git 触发重启

推送一个空提交来触发重建：

```bash
cd /Users/anthony/Desktop/llm/foodAIAgent
git commit --allow-empty -m "Trigger rebuild"
git push huggingface main
```

### 方法 3：清除缓存并重启

1. 在 Space 设置页面
2. 点击 **"Clear cache"**（如果有）
3. 然后点击 **"Restart"**

## 验证重启成功

重启后，检查以下内容：

1. **Logs 标签页**:
   - 应该看到新的构建日志
   - 显示 "Building..." 然后 "Running"

2. **应用界面**:
   - 刷新浏览器页面
   - 检查 UI 是否为英文
   - 测试查询功能

3. **关键特征**:
   - Settings 侧边栏显示英文
   - 思考过程显示英文节点名称
   - 查询响应为英文

## 如果仍然不工作

1. **检查 Secrets 配置**:
   - 确保 `OPENAI_API_KEY` 和 `TAVILY_API_KEY` 已设置
   - 在 Settings → Repository secrets 中查看

2. **检查构建错误**:
   - 查看 Logs 标签页是否有错误信息
   - 常见问题：依赖安装失败、环境变量缺失

3. **手动重建**:
   - 在 Space 设置页面
   - 找到 "Hard refresh" 或 "Rebuild" 选项

