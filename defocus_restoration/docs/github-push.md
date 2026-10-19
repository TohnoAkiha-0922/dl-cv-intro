# 推送到现有 GitHub 仓库

仓库：https://github.com/TohnoAkiha-0922/dl-cv-intro

本地仓库：`D:\dl-cv-intro`；项目子目录：`defocus_restoration`。
当前分支：`master`，跟踪 `origin/master`。

在 PowerShell 中按顺序执行：

```powershell
cd D:\dl-cv-intro
git add -- defocus_restoration
git diff --cached --stat
git commit -m "Add optical defocus restoration project"
git push origin master
```

`git diff --cached --stat` 展示这次提交的文件。核对后执行下一行。
完成后在 GitHub 打开 `defocus_restoration/README.md` 查看说明和结果图。

仓库当前还有先前整理到“练习”目录的文件改动。这组命令的 `git add` 选择本次新增项目；其他改动可另行整理提交。

源代码、文档和两张 README 示例图纳入版本管理。`outputs/`、模型权重、Python 缓存和本地内阁会话导出由 `.gitignore` 过滤，实验输出与权重已随项目复制保留在本地。

以后修改本项目后，也可以按同样步骤提交，将 commit 信息改成本次具体改动。
