# 安全说明

本文记录本仓库处理凭据、模型文件和生成产物的默认规则。

## GitHub 认证

请使用 GitHub CLI 的标准登录流程：

```bash
gh auth login
```

推荐选择 GitHub.com、HTTPS，并按提示用浏览器或一次性 token 登录。不要把 GitHub PAT 写入：

- git remote URL
- `.git/config`
- `.env`
- shell 脚本
- README、JSONL 示例或 issue 模板

如果 token 曾经出现在聊天、终端回显、截图或文档里，建议在 GitHub 设置中立即撤销并重新生成。

## 远端地址

推荐 remote 形态：

```bash
git remote add origin https://github.com/<owner>/qwen3-tts-openvino.git
```

不要使用这种形态：

```text
https://<token>@github.com/<owner>/qwen3-tts-openvino.git
```

## 提交前检查

确认没有凭据或大文件被加入暂存区：

```bash
git diff --cached --name-only
git ls-files | rg '^(models|openvino|openvino_full|outputs)/|\\.bin$|\\.wav$'
rg -n 'ghp_|github_pat_|token|password|secret' \
  --glob '!models/**' \
  --glob '!openvino*/**' \
  --glob '!.venv/**' \
  --glob '!.uv-cache/**' .
```

第一条用于人工检查文件列表；第二条应无输出；第三条如果命中文档中的安全说明，需要确认没有真实密钥。

## 模型和音频

模型权重、OpenVINO IR、参考音频和生成音频都可能包含授权或隐私风险。默认将这些内容保留在本地，并通过 `.gitignore` 排除。
