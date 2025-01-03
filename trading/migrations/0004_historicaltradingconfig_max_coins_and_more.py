# Generated by Django 5.1.4 on 2024-12-29 13:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trading', '0003_historicaltradingconfig_max_amount_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='historicaltradingconfig',
            name='max_coins',
            field=models.PositiveSmallIntegerField(default=2, help_text='Maximum number of coins to recommend'),
        ),
        migrations.AddField(
            model_name='historicaltradingconfig',
            name='min_coins',
            field=models.SmallIntegerField(default=1, help_text='Minimum number of coins to recommend (0 means no minimum)'),
        ),
        migrations.AddField(
            model_name='tradingconfig',
            name='max_coins',
            field=models.PositiveSmallIntegerField(default=2, help_text='Maximum number of coins to recommend'),
        ),
        migrations.AddField(
            model_name='tradingconfig',
            name='min_coins',
            field=models.SmallIntegerField(default=1, help_text='Minimum number of coins to recommend (0 means no minimum)'),
        ),
    ]
