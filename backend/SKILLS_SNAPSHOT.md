<skills>
  <summary>Available local skills that the agent can inspect with read_file.</summary>
  <skill name="代码修改" path="skills/code-modification/SKILL.md">
    <description>面向本地代码仓库的精确修改任务：先读懂上下文与调用链，再按最小改动原则用 edit_file 精确替换或 write_file 新建文件，改完做语法与静态自检并复查差异。适用于修复缺陷、实现小功能、重构局部代码、调整配置与脚本等场景。动手前必须先用 read_file/grep 建立"问题→代码位置"映射。</description>
  </skill>
  <skill name="代码开发" path="skills/coding-skill/SKILL.md">
    <description>处理代码开发类任务，包括阅读与分析代码、定位 Bug、修改代码、运行测试与代码验证。适用于用户要求解释某段代码、排查报错或 Bug、修改或新增功能、运行测试、检查语法、重构代码、编写脚本等与项目代码直接相关的场景。执行时复用现有 ReAct Agent，通过 read_file、edit_file、write_file、terminal、python_repl 等已有工具围绕项目代码目录工作。</description>
  </skill>
  <skill name="天气查询" path="skills/get_weather/SKILL.md">
    <description>查询指定城市的天气情况，并整理成适合直接回复用户的简洁结果。</description>
  </skill>
  <skill name="Git 管理" path="skills/git-management/SKILL.md">
    <description>面向本地 git 仓库的版本管理：查看状态、差异、提交历史与分支，暂存与提交改动，切换或新建分支。适用于用户要求查看改动、整理并提交、写提交信息、查看历史、对比差异、切分支等场景。只提交不推送；删除分支等破坏性操作必须先确认。</description>
  </skill>
  <skill name="日志分析" path="skills/log-analysis/SKILL.md">
    <description>面向本地日志文件的定位与分析任务：按级别和关键字过滤日志、统计错误分布、还原报错时间线与异常堆栈、聚合错误指纹并给出根因线索和证据行号。适用于排查报错、分析运行日志、定位异常请求、统计错误频次、从日志反推代码问题等场景。先用 log_stats 看全局，再用 log_search 精确定位，必要时用 grep 跨文件关联、用 python_repl 做聚合统计。</description>
  </skill>
  <skill name="kb-retriever" path="skills/rag-skill/SKILL.md">
    <description>面向本地知识库目录的检索和问答助手。核心流程：(1)分层索引导航 (2)遇到PDF/Excel时必须先读取references学习处理方法 (3)处理文件后再检索。按文件类型组合使用 grep、Read、pdfplumber、pandas 进行渐进式检索，避免整文件加载。用户问题涉及"从知识库目录回答问题/检索信息/查资料"时使用。</description>
  </skill>
  <skill name="失败恢复经验沉淀" path="skills/retry-lesson-capture/SKILL.md">
    <description>当一个任务首次执行失败，但在重试其他工具、接口、参数或流程后成功时，使用此技能总结可复用经验，并将经验写入当前正在使用的 SKILL.md 与长期记忆库（Memory API）。适用于 API 失败后切换备用 API、命令失败后改用其他命令、抓取失败后改用其他数据源、解析失败后改用其他流程等场景。</description>
  </skill>
  <skill name="测试编写" path="skills/test-authoring/SKILL.md">
    <description>为已有代码补充单元测试或回归测试：先读懂被测函数的行为契约与边界条件，再按项目现有测试框架与风格编写最小、可读、可验收的用例，最后运行验证。适用于用户要求补测试、提高覆盖率、为已修复的 Bug 补回归用例等场景。</description>
  </skill>
  <skill name="测试执行与验证" path="skills/test-execution/SKILL.md">
    <description>运行既有测试、复现问题、验证代码改动是否达标。自动探测 pytest/unittest/npm/vitest/jest/go/cargo，输出退出码、通过/失败统计与失败定位；也用于构造最小复现脚本验证假设。适用于用户要求跑测试、验证改动、复现 bug、检查回归等场景。</description>
  </skill>
  <skill name="联网搜索" path="skills/web-search/SKILL.md">
    <description>使用 Tavily 联网搜索最新信息、官方文档、新闻动态、实时行情和外部事实来源。适用于用户明确要求搜索、联网、查官网、给链接、核验事实，或任务明显依赖实时外部信息的场景。优先调用本技能目录下的 Tavily 脚本，不要退回抓搜索结果页。</description>
  </skill>
</skills>
