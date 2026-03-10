# DLRover Dashboard Static Files

## Overview
虽然仪表盘主要使用 CDN 加载前端依赖（Vue.js、Tailwind CSS、ECharts），但我们添加了静态文件以提供：

1. **可自定义的样式** - `style.css` 包含自定义主题和组件样式
2. **本地化功能** - `dashboard.js` 包含实用函数，可离线使用或扩展功能
3. **加载体验** - `loader.js` 在页面加载时提供更好的用户体验
4. **品牌标识** - 支持 favicon.ico 和自定义品牌元素

## Static File Structure
```
static/
├── css/
│   └── style.css          # 自定义样式和主题
├── js/
│   ├── dashboard.js       # 仪表盘实用函数和图表工具
│   └── loader.js          # 页面加载动画
└── img/
    └── favicon.ico        # 网站图标（需要替换为实际的 .ico 文件）
```

## 使用 CDN + 静态文件的好处

### 1. 性能优势
- CDN 提供全球分发，加速资源加载
- 静态文件用于自定义内容和本地化资源

### 2. 可定制性
- 企业可以将静态文件替换为自己的品牌和样式
- 支持离线部署（替换 CDN 链接为本地文件）

### 3. 可扩展性
- dashboard.js 为将来的功能扩展提供了框架
- 易于集成更多的图表和可视化组件

## 自定义样式

在 `style.css` 中可以自定义：
- 状态指示器的颜色（运行、失败、待机等）
- 节点类型标签的样式
- 滚动条的外观
- 响应式设计的断点
- 动画效果

## JavaScript 实用函数

`dashboard.js` 提供了：
- `DashboardUtils` - 格式化函数、颜色工具
- `ChartUtils` - ECharts 图表初始化工具
- `AutoRefresher` - 自动刷新管理器
- `DashboardWebSocket` - WebSocket 连接管理类

## 如何完全本地化

如果需要完全本地化（不依赖 CDN），可以：

1. 下载依赖包并放入 static 目录：
   ```bash
   # 下载 Vue.js
   curl -O https://unpkg.com/vue@3/dist/vue.global.js -o static/js/vue.global.js

   # 下载 Tailwind CSS（建议使用构建版本）
   curl -O https://unpkg.com/tailwindcss/dist/tailwind.min.css -o static/css/tailwind.min.css

   # 下载 ECharts
   curl -O https://cdn.jsdelivr.net/npm/echarts@5.4/dist/echarts.min.js -o static/js/echarts.min.js
   ```

2. 更新 index.html 中的 CDN 引用为本地文件。

3. 如果使用 Tailwind 构建版本，你可以自定义配置。

## 未来扩展

静态文件结构支持：
- 更多的自定义图表样式
- 主题切换功能
- 本地化语言文件
- 离线模式支持
- 自定义品牌资产（logo、图标等）