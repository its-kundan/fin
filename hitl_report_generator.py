"""
HITL Report Generator - Human-in-the-Loop Report Generation
Allows human review and editing of report sections before final generation
"""

import re
import os
import base64
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from hitl_state import HITLStateManager, Checkpoint, ApprovalStatus
import streamlit as st


class HITLReportGenerator:
    """Report generator with HITL checkpoints for section review and editing"""
    
    def __init__(self, hitl_manager: HITLStateManager):
        self.hitl = hitl_manager
        self.base_dir = Path(__file__).parent
        self.chart_directory = self.base_dir / "charts"
        self.charts_html_directory = self.base_dir / "charts_html"
    
    def generate_report_structure(self, approved_content: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate report structure from approved content.
        Returns structured report data for human review.
        """
        # Extract approved insights
        approved_insights = approved_content.get('insights', [])
        approved_charts = approved_content.get('charts', [])
        
        # Structure report sections
        report_sections = []
        
        # Section 1: Executive Summary (from approved insights)
        executive_summary = self._extract_executive_summary(approved_insights)
        report_sections.append({
            'id': 'executive_summary',
            'title': 'Executive Summary',
            'content': executive_summary,
            'editable': True,
            'type': 'text'
        })
        
        # Section 2: Key Insights (from approved insights)
        key_insights = self._extract_key_insights(approved_insights)
        report_sections.append({
            'id': 'key_insights',
            'title': 'Key Insights',
            'content': key_insights,
            'editable': True,
            'type': 'list'
        })
        
        # Section 3: Chart Catalog (from approved charts)
        chart_catalog = self._build_chart_catalog(approved_charts, approved_insights)
        report_sections.append({
            'id': 'chart_catalog',
            'title': 'Chart Catalog',
            'content': chart_catalog,
            'editable': True,
            'type': 'charts'
        })
        
        # Section 4: Recommendations (from approved insights)
        recommendations = self._extract_recommendations(approved_insights)
        report_sections.append({
            'id': 'recommendations',
            'title': 'Data-Driven Recommendations',
            'content': recommendations,
            'editable': True,
            'type': 'list'
        })
        
        return {
            'sections': report_sections,
            'metadata': {
                'dataset_name': state.get('dataset_info', {}).get('name', 'Dataset'),
                'total_charts': len(approved_charts),
                'total_insights': len(approved_insights),
                'generated_at': self._get_timestamp()
            }
        }
    
    def _extract_executive_summary(self, insights: List[Dict[str, Any]]) -> str:
        """Extract executive summary from approved insights"""
        summary_parts = []
        
        for insight in insights:
            content = insight.get('content', '')
            if '[GEMINI INSIGHTS]' in content or 'EXECUTIVE SUMMARY' in content:
                # Extract summary section from Gemini insights
                summary_match = re.search(
                    r'####\s*1\.\s*Overall Performance Summary.*?\n(.*?)(?=\n####|\Z)',
                    content,
                    re.DOTALL | re.IGNORECASE
                )
                if summary_match:
                    summary_parts.append(summary_match.group(1).strip())
        
        if not summary_parts:
            # Fallback: use first insight as summary
            if insights:
                summary_parts.append(insights[0].get('content', '')[:500])
        
        return '\n\n'.join(summary_parts) if summary_parts else "Executive summary will be generated from approved insights."
    
    def _extract_key_insights(self, insights: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights as a list"""
        key_insights = []
        
        for insight in insights:
            content = insight.get('content', '')
            if '[GEMINI INSIGHTS]' in content:
                # Extract key insights section
                insights_match = re.search(
                    r'####\s*2\.\s*Key Insights.*?\n(.*?)(?=\n####|\Z)',
                    content,
                    re.DOTALL | re.IGNORECASE
                )
                if insights_match:
                    insights_text = insights_match.group(1).strip()
                    # Split into bullet points
                    bullets = re.findall(r'^\s*\*\s*(.+)$', insights_text, re.MULTILINE)
                    key_insights.extend(bullets)
            elif '[LLAMA INSIGHTS]' in content:
                # Extract bullet points from Llama insights
                bullets = re.findall(r'^\s*[•\*]\s*(.+)$', content, re.MULTILINE)
                key_insights.extend(bullets)
        
        if not key_insights:
            key_insights = ["Key insights will be extracted from approved content."]
        
        return key_insights
    
    def _extract_recommendations(self, insights: List[Dict[str, Any]]) -> List[str]:
        """Extract recommendations from approved insights"""
        recommendations = []
        
        for insight in insights:
            content = insight.get('content', '')
            if '[GEMINI INSIGHTS]' in content:
                # Extract recommendations section
                rec_match = re.search(
                    r'####\s*3\.\s*Data-Driven Recommendations.*?\n(.*?)(?=\n####|\Z)',
                    content,
                    re.DOTALL | re.IGNORECASE
                )
                if rec_match:
                    rec_text = rec_match.group(1).strip()
                    # Split into numbered recommendations
                    items = re.findall(r'^\s*\d+\.\s*(.+)$', rec_text, re.MULTILINE)
                    recommendations.extend(items)
        
        if not recommendations:
            recommendations = ["Recommendations will be generated from approved insights."]
        
        return recommendations
    
    def _build_chart_catalog(self, approved_charts: List[Dict[str, Any]], insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build chart catalog from approved charts and insights"""
        chart_catalog = []
        
        # Extract chart explanations from Gemini insights
        chart_explanations = {}
        for insight in insights:
            content = insight.get('content', '')
            if '[GEMINI INSIGHTS]' in content:
                # Extract chart catalog section
                catalog_match = re.search(
                    r'####\s*0\.\s*Chart Catalog.*?\n(.*?)(?=\n####|\Z)',
                    content,
                    re.DOTALL | re.IGNORECASE
                )
                if catalog_match:
                    catalog_text = catalog_match.group(1)
                    # Parse chart entries
                    chart_entries = re.findall(
                        r'\d+\.\s*\*\*(.*?)\*\*:\s*(.*?)(?=\n\d+\.|\Z)',
                        catalog_text,
                        re.DOTALL
                    )
                    for chart_name, explanation in chart_entries:
                        # Extract filename
                        filename_match = re.search(r'([\w.-]+\.(png|jpg|jpeg))', chart_name, re.IGNORECASE)
                        if filename_match:
                            filename = filename_match.group(1)
                            chart_explanations[filename] = explanation.strip()
        
        # Build catalog from approved charts
        for i, chart in enumerate(approved_charts):
            chart_path = chart.get('path', '')
            chart_filename = Path(chart_path).name if chart_path else f"chart_{i}.png"
            
            explanation = chart_explanations.get(chart_filename, 
                f"Chart analysis for {chart.get('title', 'Chart')}")
            
            chart_catalog.append({
                'id': chart.get('id', f"chart_{i}"),
                'filename': chart_filename,
                'title': chart.get('title', f"Chart {i+1}"),
                'path': chart_path,
                'explanation': explanation,
                'editable': True
            })
        
        return chart_catalog
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def generate_html_report(self, report_structure: Dict[str, Any], 
                            edited_sections: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate final HTML report from approved and edited sections.
        Uses edited sections if provided, otherwise uses original content.
        """
        sections = report_structure.get('sections', [])
        metadata = report_structure.get('metadata', {})
        
        # Apply edits if provided
        if edited_sections:
            for section in sections:
                section_id = section['id']
                if section_id in edited_sections:
                    section['content'] = edited_sections[section_id]
        
        # Build HTML
        nav_html = ""
        content_html = ""
        
        for i, section in enumerate(sections):
            section_id = section['id']
            title = section['title']
            content = section['content']
            section_type = section.get('type', 'text')
            
            active_class = "active" if i == 0 else ""
            display_style = 'display: block;' if i == 0 else 'display: none;'
            
            nav_html += f"""
            <button class="tablinks {active_class}" onclick="openSection(event, '{section_id}')">
                {title}
            </button>
            """
            
            # Format content based on type
            if section_type == 'list':
                formatted_content = self._format_list_to_html(content)
            elif section_type == 'charts':
                formatted_content = self._format_charts_to_html(content)
            else:
                formatted_content = self._format_text_to_html(content)
            
            content_html += f"""
            <div id="{section_id}" class="tabcontent" style="{display_style}">
                <div class="report-section">
                    <h3>{title}</h3>
                    {formatted_content}
                </div>
            </div>
            """
        
        # Generate full HTML
        html_template = self._get_html_template(nav_html, content_html, metadata)
        
        return html_template
    
    def _format_text_to_html(self, text: str) -> str:
        """Format text content to HTML"""
        # Convert markdown to HTML
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = text.replace('\n\n', '</p><p>')
        text = text.replace('\n', '<br>')
        return f"<p>{text}</p>"
    
    def _format_list_to_html(self, items: List[str]) -> str:
        """Format list items to HTML"""
        list_html = ""
        for item in items:
            # Clean up item
            item = item.strip().lstrip('•*').strip()
            list_html += f"<li>{item}</li>"
        return f"<ul class='key-points-list'>{list_html}</ul>"
    
    def _format_charts_to_html(self, chart_catalog: List[Dict[str, Any]]) -> str:
        """Format chart catalog to HTML"""
        charts_html = "<div class='sequential-chart-list'>"
        
        for chart in chart_catalog:
            chart_path = chart.get('path', '')
            chart_filename = chart.get('filename', '')
            title = chart.get('title', 'Chart')
            explanation = chart.get('explanation', '')
            
            # Try to load chart image
            img_tag = ""
            if chart_path and Path(chart_path).exists():
                try:
                    img_base64 = self._image_to_base64(Path(chart_path))
                    if img_base64:
                        img_tag = f'<img src="{img_base64}" alt="{title}" style="max-width: 100%; height: auto;">'
                except Exception as e:
                    img_tag = f'<div class="chart-placeholder">Error loading chart: {e}</div>'
            else:
                img_tag = f'<div class="chart-placeholder">Chart not found: {chart_filename}</div>'
            
            charts_html += f"""
            <div class="chart-item">
                <h5>{title}</h5>
                <div class="chart-visual-wrapper">{img_tag}</div>
                <div class="chart-explanation">
                    <p>{explanation}</p>
                </div>
            </div>
            """
        
        charts_html += "</div>"
        return charts_html
    
    def _image_to_base64(self, file_path: Path) -> Optional[str]:
        """Convert image to base64"""
        try:
            if not file_path.is_file():
                return None
            with open(file_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                mime_type = f"image/{file_path.suffix.lstrip('.')}"
                return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            print(f"Error converting {file_path} to base64: {e}")
            return None
    
    def _get_html_template(self, nav_html: str, content_html: str, metadata: Dict[str, Any]) -> str:
        """Get HTML template with styling"""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Intelligent Data Analysis Report - {metadata.get('dataset_name', 'Dataset')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f7f6; }}
        .container {{ width: 95%; margin: 20px auto; background: #fff; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-radius: 8px; }}
        h1 {{ padding: 20px; margin: 0; background-color: #3f51b5; color: white; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
        .tab {{ overflow: hidden; border-bottom: 1px solid #ccc; background-color: #e9e9e9; display: flex; }}
        .tab button {{ background-color: inherit; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 15px; border-right: 1px solid #ddd; flex-grow: 1; min-width: 150px; text-align: left; }}
        .tab button:hover {{ background-color: #ddd; }}
        .tab button.active {{ background-color: #ccc; font-weight: bold; }}
        .tabcontent {{ display: none; padding: 20px; border-top: none; min-height: 600px; }}
        .report-section {{ width: 100%; overflow-y: auto; }}
        .key-points-list {{ list-style-type: none; padding-left: 0; margin: 0; }}
        .key-points-list li {{ margin-bottom: 10px; line-height: 1.4; padding-left: 25px; text-indent: -25px; }}
        .key-points-list li::before {{ content: "•"; color: #ff9800; font-weight: bold; display: inline-block; width: 25px; }}
        .sequential-chart-list {{ display: flex; flex-direction: column; gap: 40px; margin-top: 20px; padding: 15px; background-color: #fcfcfc; border: 1px solid #eee; border-radius: 8px; }}
        .chart-item {{ border: 1px solid #e0e0e0; padding: 20px; border-radius: 8px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px; }}
        .chart-item h5 {{ margin: 0 0 15px 0; color: #2e7d32; border-bottom: 1px dashed #eee; padding-bottom: 5px; font-size: 1.1em; }}
        .chart-visual-wrapper {{ width: 100%; text-align: center; margin-bottom: 15px; }}
        .chart-visual-wrapper img {{ max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px; }}
        .chart-explanation p {{ font-size: 0.95em; line-height: 1.5; margin-top: 5px; color: #333; }}
        .chart-placeholder {{ color: #cc0000; border: 1px dashed #cc0000; padding: 10px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Intelligent Data Analysis Report - {metadata.get('dataset_name', 'Dataset')}</h1>
        <div class="tab">
            {nav_html}
        </div>
        {content_html}
    </div>
    <script>
        function openSection(evt, sectionId) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(sectionId).style.display = "block";
            evt.currentTarget.className += " active";
        }}
        document.addEventListener('DOMContentLoaded', function() {{
            var activeTab = document.querySelector('.tablinks.active');
            if (activeTab) {{
                activeTab.click();
            }}
        }});
    </script>
</body>
</html>
        """

