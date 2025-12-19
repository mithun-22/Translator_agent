from django import forms
from .utils import LANGUAGE_SLUGS

ENGINE_CHOICES = [
    ("google", "Google Translator"),
    ("gemini", "Gemini (LLM-based)"),
]

class TranslationForm(forms.Form):
    source_text = forms.CharField(
        required=False, 
        widget=forms.Textarea(attrs={
            'rows': 6, 
            'cols': 50, 
            'style': 'overflow-y: auto; min-height: 150px;', 
            'placeholder': 'Enter source text or upload a PDF file below...',
            'class': 'form-control'
        })
    )
    
    source_lang = forms.ChoiceField(
        label='Source language', 
        choices=list(LANGUAGE_SLUGS.items()),
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    target_lang = forms.ChoiceField(
        label='Target language', 
        choices=list(LANGUAGE_SLUGS.items()),
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    source_file = forms.FileField(
        required=False, 
        label='Upload PDF file',
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'accept': '.pdf'
        }),
        help_text='Upload a PDF file (max 10 MB). Advanced processing with OCR support.'
    )
    
    engine = forms.ChoiceField(
        choices=ENGINE_CHOICES, 
        required=True, 
        initial="gemini",
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        help_text='Gemini provides better quality for complex texts and formatting.'
    )

    def clean(self):
        cleaned_data = super().clean()
        source_text = cleaned_data.get('source_text', '').strip()
        source_file = cleaned_data.get('source_file')
        source_lang = cleaned_data.get('source_lang')
        target_lang = cleaned_data.get('target_lang')

        # Basic validation - detailed validation happens in PDF2text.py
        if not source_text and not source_file:
            raise forms.ValidationError("Please provide either text or upload a PDF file.")
        
        if source_text and source_file:
            raise forms.ValidationError("Please provide either text OR a PDF file, not both.")
        
        if source_lang == target_lang:
            raise forms.ValidationError("Source and target languages must be different.")
        
        # Basic file validation
        if source_file:
            if not source_file.name.lower().endswith('.pdf'):
                raise forms.ValidationError("Only PDF files are allowed.")
            
            max_size = 10 * 1024 * 1024  # 10 MB
            if source_file.size > max_size:
                raise forms.ValidationError("File size must be under 10 MB.")

        return cleaned_data
