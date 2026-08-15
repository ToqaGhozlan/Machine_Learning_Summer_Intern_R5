"""
Django forms for weather prediction.
"""

from django import forms


class TemperaturePredictionForm(forms.Form):
    """
    Form for temperature prediction input.
    
    Users specify how many days ahead they want to forecast.
    """
    
    HORIZON_CHOICES = [
        (1, '1 day ahead'),
        (2, '2 days ahead'),
        (3, '3 days ahead'),
        (4, '4 days ahead'),
        (5, '5 days ahead'),
        (7, '1 week ahead'),
        (14, '2 weeks ahead'),
        (21, '3 weeks ahead'),
        (28, '4 weeks ahead'),
    ]
    
    horizon_days = forms.ChoiceField(
        choices=HORIZON_CHOICES,
        # ChoiceField values are submitted as strings.  Keeping the initial
        # value a string lets the template mark the first option as checked.
        initial='1',
        widget=forms.RadioSelect,
        label='Forecast Horizon',
        help_text='Select how many days ahead you want to predict temperature'
    )
    
    class Meta:
        fields = ['horizon_days']
    
    def clean_horizon_days(self):
        """Validate horizon_days"""
        horizon = self.cleaned_data.get('horizon_days')
        try:
            horizon_int = int(horizon)
            if horizon_int < 1 or horizon_int > 28:
                raise forms.ValidationError(
                    'Forecast horizon must be between 1 and 28 days.'
                )
            return horizon_int
        except (ValueError, TypeError):
            raise forms.ValidationError('Invalid horizon value.')
