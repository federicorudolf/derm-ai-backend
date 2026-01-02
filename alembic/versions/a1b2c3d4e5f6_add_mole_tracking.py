"""add_mole_tracking

Revision ID: a1b2c3d4e5f6
Revises: 9e1be52501da
Create Date: 2026-01-02 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '9e1be52501da'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create moles table
    op.create_table('moles',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=True),
    sa.Column('body_part_location', sa.String(), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('is_archived', sa.Boolean(), nullable=False, server_default='false'),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_moles_id'), 'moles', ['id'], unique=False)

    # Add mole_id column to pictures table
    op.add_column('pictures', sa.Column('mole_id', sa.Integer(), nullable=True))
    op.create_foreign_key('fk_pictures_mole_id', 'pictures', 'moles', ['mole_id'], ['id'])


def downgrade() -> None:
    # Remove mole_id from pictures table
    op.drop_constraint('fk_pictures_mole_id', 'pictures', type_='foreignkey')
    op.drop_column('pictures', 'mole_id')

    # Drop moles table
    op.drop_index(op.f('ix_moles_id'), table_name='moles')
    op.drop_table('moles')
