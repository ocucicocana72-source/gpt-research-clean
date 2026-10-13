import fs from 'node:fs/promises';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const outDir = '/Users/ernurat/Desktop/agent学习/gpt_research/学习计划';
await fs.mkdir(outDir, { recursive: true });

const wb = Workbook.create();
const summary = wb.worksheets.add('计划总览');
const daily = wb.worksheets.add('每日计划');
summary.showGridLines = false;
daily.showGridLines = false;
summary.tabColor = '#1F4E78';
daily.tabColor = '#70AD47';

const navy = '#1F4E78';
const blue = '#D9EAF7';
const lightBlue = '#EAF3F8';
const green = '#E2F0D9';
const orange = '#FCE4D6';
const gray = '#F2F2F2';
const dark = '#1F2937';
const white = '#FFFFFF';
const border = '#B7C9D6';

// Summary sheet
summary.mergeCells('A1:H1');
summary.getRange('A1').values = [['GPT Researcher 源码学习与 Agent 改造计划']];
summary.getRange('A1:H1').format = { fill: navy, font: { bold: true, color: white, size: 16 }, horizontalAlignment: 'center', verticalAlignment: 'center' };
summary.getRange('A1:H1').format.rowHeight = 32;
summary.mergeCells('A2:H2');
summary.getRange('A2').values = [['周期：2026/10/03–2026/10/08　｜　目标：读懂主线、完成一次改造、对照学习一个轻量 Agent 项目']];
summary.getRange('A2:H2').format = { fill: lightBlue, font: { color: dark, italic: true }, horizontalAlignment: 'center', verticalAlignment: 'center' };

summary.getRange('A4:B7').values = [
  ['学习目标', '完成标准'],
  ['源码理解', '能脱离源码讲清 query → 检索 → context → report'],
  ['项目改造', '有一个可运行的中文技术/论文调研 Agent 改版'],
  ['迁移能力', '能解释 GPT Researcher 与 mini-SWE-agent 的 Agent 循环差异'],
];
summary.getRange('A4:B4').format = { fill: navy, font: { bold: true, color: white }, horizontalAlignment: 'center' };
summary.getRange('A5:B7').format = { fill: white, font: { color: dark }, wrapText: true, verticalAlignment: 'center' };
summary.getRange('A4:B7').format.borders = { preset: 'all', style: 'thin', color: border };

summary.getRange('D4:E7').values = [
  ['总体进度', '数值'],
  ['已完成天数', '=COUNTIF(\'每日计划\'!$H$5:$H$16,"已完成")'],
  ['计划天数', '=COUNTA(\'每日计划\'!$A$5:$A$16)'],
  ['完成率', '=IF(E6=0,0,E5/E6)'],
];
summary.getRange('D4:E4').format = { fill: navy, font: { bold: true, color: white }, horizontalAlignment: 'center' };
summary.getRange('D5:E7').format = { fill: white, font: { color: dark }, verticalAlignment: 'center' };
summary.getRange('D4:E7').format.borders = { preset: 'all', style: 'thin', color: border };
summary.getRange('E7').format.numberFormat = '0%';
summary.getRange('E5:E7').format.horizontalAlignment = 'center';

summary.getRange('A10:H16').values = [
  ['日期', '阶段', '核心目标', '主要文件 / 资料', '实践任务', '当天产出', '完成标准', '状态'],
  ['10/03', '源码理解 1/2', '读懂单 Agent 主骨架', 'README.md；cli.py；gpt_researcher/agent.py；skills/researcher.py；skills/writer.py', '画出一次研究请求的调用图', '主链路图 + 关键对象说明', '能解释 query、context、report 的流转', '未开始'],
  ['10/04', '源码理解 2/2', '补齐检索、抓取、上下文和配置机制', 'actions/query_processing.py；actions/retriever.py；skills/browser.py；skills/context_manager.py；actions/report_generation.py；config/config.py', '为搜索为空、抓取失败、上下文为空分别写排查路径', '一页项目结构笔记', '能定位主要问题的排查入口', '未开始'],
  ['10/05', '项目改造 1/2', '确定并实现自己的 Agent 方向', '沿用核心 Agent 主线；优先改 Prompt、检索源或报告结构', '选择两个真实问题运行原版与改版', '改版第一版 + 两组输出', '改版能从输入问题生成报告', '未开始'],
  ['10/06', '项目改造 2/2', '验证、整理并形成可演示项目', 'tests/；README；输出报告', '增加“资料不足”问题，检查引用、失败表现和成本', 'README + 演示流程 + 验证记录', '能现场演示并解释与原版的差异', '未开始'],
  ['10/07', '对照学习', '学习 mini-SWE-agent 的轻量 Agent 循环', 'run/mini.py；agents/default.py；models/；environments/local.py；官方 control_flow 文档', '画出模型查询、执行动作、观察回填、终止的循环', '对照图 + 半页比较笔记', '能比较两种 Agent 的任务、工具、状态和终止条件', '未开始'],
  ['10/08', '复盘与机动', '补缺口并形成自己的理解', 'deep_research.py；multi_agents/agents/orchestrator.py；自己的改版', '脱离源码做 5 分钟口头讲解', '项目结构图 + 可运行改版 + 讲解提纲', '能独立说明设计取舍和下一步', '未开始'],
];
summary.getRange('A10:H10').format = { fill: navy, font: { bold: true, color: white }, horizontalAlignment: 'center', verticalAlignment: 'center', wrapText: true };
summary.getRange('A11:H16').format = { fill: white, font: { color: dark }, wrapText: true, verticalAlignment: 'center' };
summary.getRange('A10:H16').format.borders = { preset: 'all', style: 'thin', color: border };
summary.getRange('H11:H16').dataValidation = { rule: { type: 'list', values: ['未开始', '进行中', '已完成', '延期'] } };
summary.getRange('H11:H16').conditionalFormats.add('containsText', { text: '已完成', format: { fill: green, font: { color: '#2F6B2F', bold: true } } });
summary.getRange('H11:H16').conditionalFormats.add('containsText', { text: '进行中', format: { fill: blue, font: { color: '#1F4E78', bold: true } } });
summary.getRange('H11:H16').conditionalFormats.add('containsText', { text: '延期', format: { fill: orange, font: { color: '#9C0006', bold: true } } });

summary.getRange('A19:H22').values = [
  ['每日固定节奏', '建议时长', '执行方式', '验收问题', '', '', '', ''],
  ['源码阅读', '3 小时', '先看入口，再沿调用关系向下读；记录输入、输出、是否调用网络或 LLM', '今天能否不用看代码复述主流程？', '', '', '', ''],
  ['动手实践', '2 小时', '跑一个小例子、画图、改一处代码或比较一次输出', '是否留下了可运行产物或验证记录？', '', '', '', ''],
  ['复述复盘', '30 分钟', '合上源码，用自己的话解释；把说不清的点标回文件', '哪些地方仍然只能“看懂”，还不能“讲懂”？', '', '', '', ''],
];
summary.mergeCells('D19:H19'); summary.mergeCells('D20:H20'); summary.mergeCells('D21:H21'); summary.mergeCells('D22:H22');
summary.getRange('A19:H19').format = { fill: navy, font: { bold: true, color: white }, horizontalAlignment: 'center' };
summary.getRange('A20:H22').format = { fill: gray, font: { color: dark }, wrapText: true, verticalAlignment: 'center' };
summary.getRange('A19:H22').format.borders = { preset: 'all', style: 'thin', color: border };

summary.getRange('A:A').format.columnWidth = 12;
summary.getRange('B:B').format.columnWidth = 17;
summary.getRange('C:C').format.columnWidth = 24;
summary.getRange('D:D').format.columnWidth = 46;
summary.getRange('E:E').format.columnWidth = 30;
summary.getRange('F:F').format.columnWidth = 26;
summary.getRange('G:G').format.columnWidth = 31;
summary.getRange('H:H').format.columnWidth = 12;
summary.getRange('A10:H16').format.rowHeight = 68;
summary.freezePanes.freezeRows(10);

// Daily sheet
 daily.mergeCells('A1:J1');
 daily.getRange('A1').values = [['每日执行清单与复盘']];
 daily.getRange('A1:J1').format = { fill: '#548235', font: { bold: true, color: white, size: 16 }, horizontalAlignment: 'center', verticalAlignment: 'center' };
 daily.getRange('A1:J1').format.rowHeight = 32;
 daily.mergeCells('A2:J2');
 daily.getRange('A2').values = [['每天完成后填写“完成情况”和“复盘记录”，状态会回写到“计划总览”页。']];
 daily.getRange('A2:J2').format = { fill: '#E2F0D9', font: { color: dark, italic: true }, horizontalAlignment: 'center' };

daily.getRange('A4:J10').values = [
  ['日期', '时间块', '任务', '对应文件 / 路径', '输入 / 问题', '预期输出', '完成情况', '状态', '复盘记录', '下一步'],
  ['10/03', '09:00–12:00', '读入口与 GPTResearcher 主编排', 'README.md；cli.py；gpt_researcher/agent.py', '一个研究问题', '画出 GPTResearcher → ResearchConductor → ReportGenerator', '', '未开始', '', '下午继续读研究执行链'],
  ['10/03', '14:00–16:00', '读研究执行和报告生成', 'skills/researcher.py；skills/writer.py', 'query / context / report', '写出数据流转说明', '', '未开始', '', '晚上脱离源码复述'],
  ['10/04', '09:00–12:00', '读查询、检索器、抓取和上下文选择', 'actions/；skills/browser.py；skills/context_manager.py；context/', '搜索结果与网页正文', '排查路径表', '', '未开始', '', '下午补配置与 Prompt'],
  ['10/04', '14:00–16:00', '读配置、Prompt 和测试', 'config/；prompts.py；tests/', '默认配置与边界场景', '项目结构笔记', '', '未开始', '', '确定改造方向'],
  ['10/05', '09:00–12:00', '确定改版目标并实现第一版', '自己的改造文件 + 核心 Agent', '两个真实问题', '改版代码', '', '未开始', '', '下午比较原版与改版'],
  ['10/05', '14:00–16:00', '运行原版和改版并记录差异', '输出报告 / 日志', '相同输入', '对比记录', '', '未开始', '', '准备第二天验证'],
  ['10/06', '09:00–12:00', '验证正常、资料不足和异常路径', 'tests/；运行日志', '三个测试问题', '验证结果', '', '未开始', '', '下午写 README'],
  ['10/06', '14:00–16:00', '整理 README 与演示流程', 'README；项目结构图', '改版代码', '可运行项目说明', '', '未开始', '', '准备 Agent 对照学习'],
  ['10/07', '09:00–12:00', '读 mini-SWE-agent 入口和 Agent 循环', 'run/mini.py；agents/default.py', '一个简单编码任务', 'Agent 循环图', '', '未开始', '', '下午读模型和环境'],
  ['10/07', '14:00–16:00', '读模型、执行环境并对照 GPT Researcher', 'models/；environments/local.py；control_flow 文档', '两种 Agent 架构', '半页比较笔记', '', '未开始', '', '整理共性与差异'],
  ['10/08', '09:00–12:00', '补读深度研究或多 Agent 编排', 'skills/deep_research.py；multi_agents/agents/orchestrator.py', '自己的疑问清单', '补缺口笔记', '', '未开始', '', '下午做最终复盘'],
  ['10/08', '14:00–16:00', '脱离源码讲解项目并整理下一步', '自己的改版；全部笔记', '5 分钟讲解提纲', '结构图、改版、演示提纲', '', '未开始', '', '选择下一轮改进任务'],
];
daily.getRange('A4:J4').format = { fill: '#548235', font: { bold: true, color: white }, horizontalAlignment: 'center', verticalAlignment: 'center', wrapText: true };
daily.getRange('A5:J16').format = { fill: white, font: { color: dark }, wrapText: true, verticalAlignment: 'center' };
daily.getRange('A4:J16').format.borders = { preset: 'all', style: 'thin', color: border };
daily.getRange('G5:G16').format.fill = '#FFF2CC';
daily.getRange('H5:H16').dataValidation = { rule: { type: 'list', values: ['未开始', '进行中', '已完成', '延期'] } };
daily.getRange('H5:H16').conditionalFormats.add('containsText', { text: '已完成', format: { fill: green, font: { color: '#2F6B2F', bold: true } } });
daily.getRange('H5:H16').conditionalFormats.add('containsText', { text: '进行中', format: { fill: blue, font: { color: '#1F4E78', bold: true } } });
daily.getRange('H5:H16').conditionalFormats.add('containsText', { text: '延期', format: { fill: orange, font: { color: '#9C0006', bold: true } } });
daily.getRange('A:A').format.columnWidth = 11;
daily.getRange('B:B').format.columnWidth = 15;
daily.getRange('C:C').format.columnWidth = 27;
daily.getRange('D:D').format.columnWidth = 48;
daily.getRange('E:E').format.columnWidth = 22;
daily.getRange('F:F').format.columnWidth = 28;
daily.getRange('G:G').format.columnWidth = 18;
daily.getRange('H:H').format.columnWidth = 12;
daily.getRange('I:I').format.columnWidth = 28;
daily.getRange('J:J').format.columnWidth = 24;
daily.getRange('A5:J16').format.rowHeight = 48;
daily.freezePanes.freezeRows(4);

await wb.recalculate();
const inspect = await wb.inspect({ kind: 'sheet,region', sheetId: '计划总览', range: 'A1:H22', maxChars: 6000, tableMaxRows: 25, tableMaxCols: 10 });
console.log(inspect.ndjson || inspect);
const preview = await wb.render({ sheetName: '计划总览', autoCrop: 'all', scale: 1, format: 'png' });
await fs.writeFile(`${outDir}/计划总览预览.png`, new Uint8Array(await preview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(`${outDir}/GPT_Researcher_学习与改造计划.xlsx`);
console.log(`saved ${outDir}/GPT_Researcher_学习与改造计划.xlsx`);
