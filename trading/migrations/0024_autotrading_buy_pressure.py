# Generated by Django 5.1.4 on 2025-05-29 15:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trading', '0023_algorithmparameter_reasoning'),
    ]

    operations = [
        migrations.AddField(
            model_name='autotrading',
            name='buy_pressure',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
