"""Portable evidence artifacts, containing measured outputs and no generated clinical prose."""
import csv
from io import BytesIO, StringIO
import json
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape
import numpy as np
from PIL import Image as PILImage
from .images import png_bytes
from .chest import LABELS, overlay


def rgb_png(array):
    buffer = BytesIO()
    PILImage.fromarray(array).save(buffer, format='PNG')
    return buffer.getvalue()


def serializable(study):
    return {'record': study['record'], 'findings': study['rows'], 'sensitivity': study.get('sensitivity'),
            'reference_scores': study['inference']['scores'][2].tolist() if study['reference'] is not None else None}


def pdf_report(study):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, white
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='SmallX', fontSize=8, leading=11, spaceAfter=7))
    doc = SimpleDocTemplate(buffer, pagesize=(612, 792), topMargin=35, bottomMargin=38, leftMargin=40, rightMargin=40)
    record = study['record']
    story = [Paragraph('VisionX Pro | Imaging evidence', styles['Title']),
             Paragraph('RESEARCH OUTPUT - NOT A CLINICAL REPORT', styles['Heading3']),
             Paragraph(escape(record['created_utc']), styles['SmallX']),
             Paragraph('A pretrained chest X-ray model evaluated the input and restored images. Scores are uncalibrated sigmoid outputs, not disease probabilities. No diagnosis, treatment, or triage recommendation is made.', styles['BodyText']), Spacer(1,14)]
    panels = Table([[Image(BytesIO(png_bytes(study['input'])), width=170, height=170, kind='proportional'),
                     Image(BytesIO(png_bytes(study['restored'])), width=170, height=170, kind='proportional')],
                    ['Input', 'Restored: '+record['method']]], colWidths=[260,260])
    panels.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('FONTSIZE',(0,1),(-1,1),9)]))
    story += [panels, Spacer(1,12)]
    story.append(Paragraph('Finding responses',styles['Heading2']))
    rows=[['Finding','Input score','Restored score','Change']]
    for row in sorted(study['rows'],key=lambda r:-r['input_score']):
        rows.append([row['finding'], f"{row['input_score']:.4f}",f"{row['restored_score']:.4f}",f"{row['change']:+.4f}"])
    table=Table(rows,colWidths=[235,95,95,95],repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#183242')),('TEXTCOLOR',(0,0),(-1,0),white),
                              ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),
                              ('BOTTOMPADDING',(0,0),(-1,-1),2),('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#EDF3F5'),white])]))
    story.append(table)
    story += [PageBreak(),Paragraph('Provenance & interpretation',styles['Title'])]
    facts=[('Model',record['model']),('Restoration',record['method']),('Compute device',record['device']),
           ('Restoration time',f"{record['restoration_seconds']:.3f} seconds"),
           ('Paired classification time',f"{record['classification_seconds']:.3f} seconds"),
           ('Largest score shift',f"{record['largest_change_finding']}: {record['max_absolute_score_change']:.4f}"),
           ('Synthetic noise sigma',str(record['synthetic_noise_sigma'])),('Working pixel SHA-256',record['input_sha256'])]
    for label,value in facts:
        story.append(Paragraph('<b>'+escape(label)+':</b> '+escape(value),styles['SmallX']))
    target=int(np.argmax(study['inference']['scores'][0]))
    scale=max(float(np.maximum(study['inference']['cams'][:2,target],0).max()),1e-8)
    maps=[Image(BytesIO(rgb_png(overlay(study[key],study['inference']['cams'][i,target],scale))),width=210,height=210)
          for i,key in enumerate(['input','restored'])]
    story += [Paragraph('Class activation: '+LABELS[target],styles['Heading2']),Table([maps],colWidths=[260,260]),
              Paragraph('Left: input; right: restored. Shared display scale. Maps are derived from 7 x 7 features on the 224 x 224 center crop. They are class contributions, not lesion boundaries or proof of disease.',styles['SmallX'])]
    if 'reference_metrics' in record:
        m=record['reference_metrics']; d=record['reference_score_drift']
        story.append(Paragraph(f"Controlled experiment: input SSIM {m['input']['ssim']:.4f}; restored SSIM {m['restored']['ssim']:.4f}. Mean score drift from reference: input {d['input']:.4f}; restored {d['restored']:.4f}. Similarity and score consistency do not establish clinical benefit.",styles['SmallX']))
    for limitation in record['limitations']:
        story.append(Paragraph(escape(limitation),styles['SmallX']))
    story.append(Paragraph('Sources: mlmed.org/torchxrayvision | arxiv.org/abs/1512.04150 | github.com/JingyunLiang/SwinIR. Full configuration, model hashes and any sensitivity results are included in the JSON evidence export.',styles['SmallX']))
    doc.build(story)
    return buffer.getvalue()


def evidence_zip(study):
    buffer=BytesIO()
    with ZipFile(buffer,'w',ZIP_DEFLATED) as archive:
        archive.writestr('evidence.json',json.dumps(serializable(study),indent=2,allow_nan=False))
        archive.writestr('input.png',png_bytes(study['input']))
        archive.writestr('restored.png',png_bytes(study['restored']))
        if study['reference'] is not None:
            archive.writestr('reference.png',png_bytes(study['reference']))
        csv_buffer=StringIO(); writer=csv.DictWriter(csv_buffer,fieldnames=list(study['rows'][0])); writer.writeheader(); writer.writerows(study['rows'])
        archive.writestr('findings.csv',csv_buffer.getvalue())
        archive.writestr('research-report.pdf',pdf_report(study))
    return buffer.getvalue()

