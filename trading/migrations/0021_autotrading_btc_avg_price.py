# Generated by Django 5.1.4 on 2025-05-27 13:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trading', '0020_autotrading_stop_loss_signal'),
    ]

    operations = [
        migrations.AddField(
            model_name='autotrading',
            name='btc_avg_price',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
