"""Build a concise PDF and Markdown report from measured experiment outputs."""
import json
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'results/metrics.json').read_text())
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
font_path=Path('/usr/share/fonts/truetype/dejavu')
if font_path.exists():
    pdfmetrics.registerFont(TTFont('ReportSans',str(font_path/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('ReportBold',str(font_path/'DejaVuSans-Bold.ttf')))
else:
    # Built-in portable fallback for machines without DejaVu.
    pdfmetrics.registerFont(pdfmetrics.Font('ReportSans','Helvetica','WinAnsiEncoding'))
    pdfmetrics.registerFont(pdfmetrics.Font('ReportBold','Helvetica-Bold','WinAnsiEncoding'))
styles=getSampleStyleSheet()
for style in styles.byName.values(): style.fontName='ReportSans'
styles['Title'].fontName='ReportBold'
styles['Heading2'].fontName='ReportBold'
styles['Title'].fontSize=24; styles['Title'].leading=28; styles['Title'].alignment=TA_LEFT
styles['BodyText'].fontSize=9; styles['BodyText'].leading=12.5
styles['Heading2'].textColor=colors.HexColor('#087e80')
story=[]; md=[]
def title(t): story.append(Paragraph(t,styles['Title'])); md.append('# '+t+'\n')
def heading(t): story.append(Paragraph(t,styles['Heading2'])); md.append('## '+t+'\n')
def para(t): story.append(Paragraph(t,styles['BodyText'])); story.append(Spacer(1,8)); md.append(t+'\n')
def table(rows,widths):
    t=Table(rows,colWidths=widths,hAlign='LEFT'); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#14354a')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'ReportBold'),('FONTNAME',(0,1),(-1,-1),'ReportSans'),('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eef4f5'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP')]))
    story.append(t); story.append(Spacer(1,10))
    md.extend(['| '+' | '.join(map(str,rows[0]))+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows[1:]]+[''])
title('AI-powered food quality analysis')
para('Technical report | Reproducible benchmark | 15 November 2026')
heading('1. Objective and scope')
para('Predict three sensory-quality bands from laboratory measurements of wine. The research question is whether nonlinear learners improve balanced classification over simple baselines on unseen measurement profiles. This is a research-preparation benchmark, not a new algorithm or a validated food safety system. No spoilage or contamination outcome is modelled.')
heading('2. Dataset and target')
para('UCI Wine Quality [1] supplies 6,497 red and white Portuguese Vinho Verde records and 11 physicochemical inputs. Wine type is added from file identity. We remove 1,177 exact feature-plus-label duplicate rows, leaving 5,320 unique measurement profiles with no missing values. The target bands are project-defined: lower (score <=5; n=1,988), standard (6; n=2,323), and higher (>=7; n=1,009). These are not official grades. The archive, attribution, checksum and feature dictionary are included.')
heading('3. Experimental protocol')
para('Seed 42 defines the first fold of a five-fold stratified grouped partition as test: 4,256 training and 1,064 test records. Groups are hashes of input measurements, including wine type and excluding quality. This prevents exact-profile overlap. Four-fold grouped cross-validation on training data selects the model with the highest mean macro-F1. Standardization is fitted inside each logistic-regression fold. No test-driven tuning is performed.')
para('Prespecified candidates are a majority classifier, balanced logistic regression, a 300-tree random forest (minimum leaf size 2, square-root feature sampling, balanced subsample weights), and histogram gradient boosting (180 iterations, 15 leaves, L2=1). Full settings are in the source. The selected model is refitted on training data only. Test-group bootstrap resampling (1,000 replicates) estimates conditional uncertainty; it does not include training or model-selection uncertainty.')
table([['Model','CV macro-F1','Fold SD']]+[[k.replace('_',' '),f"{v['mean_macro_f1']:.4f}",f"{v['std_macro_f1']:.4f}"] for k,v in r['cross_validation'].items()],[260,110,110])
para('Primary metric: macro-F1, giving each band equal weight. Secondary metrics: accuracy, balanced accuracy and mean absolute error of band codes 0/1/2. Fold SD is descriptive, not a confidence interval.')
story.append(PageBreak())
heading('4. Held-out performance')
table([['Metric','Random forest','Majority baseline']]+[[k.replace('_',' '),f'{v:.4f}',f"{r['baseline_test_metrics'][k]:.4f}"] for k,v in r['test_metrics'].items()],[240,120,120])
para('Test macro-F1 is 0.5983 (95% conditional bootstrap interval 0.5673-0.6293). This is an absolute improvement of 0.3955 over the baseline. Accuracy is 61.94%; balanced accuracy is 58.82%. Performance is useful as a baseline, but insufficient to justify autonomous grading.')
table([['Grade','Precision','Recall','F1','Support']]+[[k,f"{r['classification_report'][k]['precision']:.3f}",f"{r['classification_report'][k]['recall']:.3f}",f"{r['classification_report'][k]['f1-score']:.3f}",str(int(r['classification_report'][k]['support']))] for k in ['lower','standard','higher']],[140,85,85,85,85])
story.append(Image(str(ROOT/'results/evaluation.png'),width=480,height=183)); md.append('![Evaluation](../results/evaluation.png)\n')
para('Higher-grade recall is 43.07%: 115 of 202 higher-grade samples are missed, including 107 classified as standard. Of 405 total errors, 15 cross both band boundaries. Red-wine macro-F1 is 0.5485 (271 test rows), versus 0.6036 for white wine (793 rows); this descriptive gap is not a significance claim. Permutation importance reports held-out predictive associations and is not used to tune the model. Correlated inputs can redistribute importance.')
heading('5. Reproducibility and verification')
para('The package includes pinned dependencies, raw data, split assignments, all held-out predictions, metrics, trained model, a prediction CLI, and this report generator. Three automated tests passed: band boundaries, dataset/split integrity, and invalid-input handling. Training and prediction were executed in Python 3.12.14 with scikit-learn 1.8.0. The CI configuration is included but has not been executed on GitHub. Run python src/build_report.py after training to rebuild this report.')
story.append(PageBreak())
heading('6. Limitations and interpretation')
para('Endpoint validity: sensory preference is not spoilage or microbiological safety. No image, pathogen, toxin or shelf-life claims are supported. Analytical banding loses information and imposes arbitrary boundaries; an ordinal alternative is a future experiment.')
para('Generalisation: this is one historical, regional wine dataset. There are no usable production-batch, winery or time identifiers in the pipeline. Deduplication prevents exact-copy leakage but cannot eliminate near-duplicate or hidden batch dependence. Removing duplicates also changes the evaluated population to unique profiles. Internal random holdout performance does not establish cross-region, cross-food or prospective performance.')
para('Statistical uncertainty: only one outer holdout and a small fixed model set were used. The bootstrap is conditional on the chosen trained model and empirical test distribution. It does not measure uncertainty from retraining, unobserved batches or dataset shift. Label subjectivity and class imbalance limit attainable agreement; minority-grade recall remains weak.')
para('Operational limits: predictions are class labels without calibrated confidence. Input validation checks column names, numeric finiteness and wine-type codes; a range flag detects only simple univariate departures from training. It cannot identify all unit mistakes, implausible combinations or distribution shifts. Human review and external validation are necessary for any operational use.')
heading('7. Research contribution and next study')
para('The contribution is a transparent and auditable implementation of established methods, including duplicate-aware evaluation and class-specific error analysis. It demonstrates research preparation rather than methodological novelty.')
para('Next, compare ordinal score models using training-only nested cross-validation; freeze the selected design before external testing. Obtain independent production batches with collection dates and repeated sensory labels. For a spoilage or contamination extension, first acquire appropriate laboratory-confirmed endpoints and sample identifiers, then use batch/time holdouts. Do not relabel sensory-quality scores as safety outcomes.')
heading('References and licensing')
para('[1] Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009). Wine Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T. Dataset license: CC BY 4.0.')
para('[2] Cortez et al. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547-553. https://doi.org/10.1016/j.dss.2009.05.016.')
para('Code license: MIT. Original dataset authors retain their attribution. No new laboratory or field collection is claimed. Full provenance and unit conventions are in docs/dataset_card.md.')
def footer(canvas,doc):
    canvas.setFont('ReportSans',8); canvas.setFillColor(colors.HexColor('#516373'))
    canvas.drawString(54,30,'FOOD QUALITY ML  /  RESEARCH BENCHMARK'); canvas.drawRightString(558,30,str(doc.page))
SimpleDocTemplate(str(ROOT/'docs/technical_report.pdf'),pagesize=(612,792),rightMargin=54,leftMargin=54,topMargin=42,bottomMargin=48).build(story,onFirstPage=footer,onLaterPages=footer)
(ROOT/'docs/technical_report.md').write_text('\n'.join(md))
