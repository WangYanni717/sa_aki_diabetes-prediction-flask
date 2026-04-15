# CatBoost 临床预测模型部署说明

这是一个基于 CatBoost 的 Flask 临床风险预测网页应用，已经整理为适合上传 GitHub 并部署到 Render 的结构。

## 仓库结构

```text
project/
├── app_flask.py
├── cat_model.pkl
├── Procfile
├── requirements.txt
├── README.md
└── .gitignore
```

## 本地运行

```bash
pip install -r requirements.txt
python app_flask.py
```

打开浏览器访问：`http://localhost:5001`

## 上传到 GitHub

1. 在项目目录初始化 Git 仓库：
   ```bash
   git init
   git add .
   git commit -m "Prepare Flask app for GitHub and Render deployment"
   ```
2. 在 GitHub 新建一个空仓库。
3. 按 GitHub 页面提示执行：
   ```bash
   git remote add origin <你的仓库地址>
   git branch -M main
   git push -u origin main
   ```

## Render 部署

1. 登录 Render，点击 **New +** → **Web Service**。
2. 连接刚刚推送到 GitHub 的仓库。
3. 配置如下：
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app_flask:app --bind 0.0.0.0:$PORT`
4. 点击 Deploy。

## 部署要点

- `app_flask.py` 会优先读取根目录的 `cat_model.pkl`。
- 启动时会自动使用 Render 注入的 `PORT` 环境变量。
- `Procfile` 已补齐，方便平台识别启动方式。

## 常见问题

- **模型加载失败**：确认 `cat_model.pkl` 已提交到仓库根目录。
- **依赖安装失败**：检查 Render 使用的 Python 版本是否兼容 `requirements.txt`。
- **页面打不开**：查看 Render 日志，确认 `gunicorn` 已成功启动并绑定到 `$PORT`。

## 说明

如果你后面还想继续整理成更标准的项目结构，比如 `src/`、`models/`、`templates/`，我也可以继续帮你拆分。
